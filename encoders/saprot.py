"""SaProt protein encoder (structure-aware, Foldseek 3Di + sequence).

Two classes with different jobs:

  SaProtModel            wraps the HuggingFace checkpoint and runs it. Used by
                         scripts/precompute_saprot.py on the GPU box. Not used
                         at training time.

  SaProtProteinEncoder   the training-time encoder. Reads precomputed
                         per-residue embeddings from cache/saprot/ and returns
                         the exact 2-tuple ProtBertProteinEncoder returns, so
                         models.py can swap one for the other at the
                         construction site and change nothing else.

SaProt consumes a structure-aware sequence built by scripts/run_foldseek.py:
one token per residue, amino acid uppercase + Foldseek 3Di lowercase ("MdKpTv"),
with '#' in the structural slot where the structure is unknown or masked. The
checkpoint's vocabulary contains those pair tokens directly (446 entries: 20x20
pairs, 21 masked forms, and the specials), so a residue is one token and the
per-residue embedding lines up with the sequence 1:1.

westlake-repl/SaProt_650M_AF2 reports hidden_size 1280 - the same width as the
ESM-2 650M baseline - so nothing downstream resizes. The projection below exists
only as a guard if a different checkpoint is ever pointed at this code.
"""

import hashlib
import os

import torch
import torch.nn as nn


# Mirrors models.py:_CACHE_ROOT - one cache subdirectory per encoder. saprot/ is
# kept apart from esm/ and esmif1/: all key on md5 of their input, so a shared
# directory would let one encoder's record answer another's lookup.
_CACHE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")

# Same truncation as ProtBertProteinEncoder (models.py:167). SaProt's own
# max_position_embeddings is 1026, so 512 is the baseline's limit, not SaProt's,
# and is kept deliberately: it makes the two encoders see the same residues.
SAPROT_MAX_LENGTH = 512

FUSION_DIM = 1280           # what the fusion stack is sized for
SAPROT_MODEL_ID = "westlake-repl/SaProt_650M_AF2"


def sequence_cache_key(sequence):
    """Same keying as models.py:_sequence_cache_key - md5 of the RAW amino-acid
    sequence, not of the SaProt structure-aware string.

    Keying on the raw sequence is what lets models.py hand this encoder exactly
    what it hands ESM-2: the dataloader never sees a SaProt sequence.
    """
    return hashlib.md5(sequence.encode("utf-8")).hexdigest()


class SaProtModel:
    """Thin wrapper over the SaProt checkpoint. Precompute-time only."""

    def __init__(self, model_path=SAPROT_MODEL_ID, device=None,
                 max_length=SAPROT_MAX_LENGTH):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.device = device if device is not None else torch.device("cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # SaProt is an ESM-architecture masked LM, so this is the same call the
        # ESM-2 path already makes (models.py:135).
        self.model = AutoModelForMaskedLM.from_pretrained(model_path).to(self.device).eval()
        self.output_dim = int(self.model.config.hidden_size)

    def encode(self, saprot_sequences):
        """SaProt sequences -> list of per-residue [L, D] tensors on cpu.

        Padding rows are dropped via the attention mask, so each returned tensor
        is that protein's real tokens only - the same unpadded-record convention
        models.py's feature cache uses (models.py:86-127).
        """
        encoded = self.tokenizer(
            saprot_sequences, padding=True, truncation=True,
            return_tensors="pt", max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True)

        features = outputs.hidden_states[-1]
        attention_mask = encoded["attention_mask"]

        results = []
        for row in range(features.shape[0]):
            keep = attention_mask[row].bool()
            results.append(features[row][keep].detach().to("cpu"))
        return results


class SaProtProteinEncoder(nn.Module):
    """Cache-backed SaProt encoder with ProtBertProteinEncoder's contract.

        forward(list[str]) -> (features [B, L_p, 1280], mask [B, L_p])

    The input is the raw amino-acid sequence, exactly what the ESM-2 encoder
    takes; the structure-aware string lives only in the precompute step.

    Each cached record is one protein:

        {"features": FloatTensor [L, D] (cpu), "length": int L,
         "has_structure": 1 if real 3Di was used, 0 if sequence-only}

    has_structure is carried for analysis - it does not change what this encoder
    returns, because a sequence-only protein was already encoded with '#' in the
    structural slots at precompute time.
    """

    def __init__(self, cache_dir=None, device=None, output_dim=FUSION_DIM):
        super().__init__()
        self.cache_dir = cache_dir if cache_dir is not None else os.path.join(_CACHE_ROOT, "saprot")
        self.device = device if device is not None else torch.device("cpu")
        self.output_dim = output_dim

        # SaProt_650M_AF2 is 1280-wide, so this is an Identity in practice and the
        # fusion stack is untouched. It is a Linear only if a narrower or wider
        # checkpoint is ever used, so the rest of CMA-DTI still sees 1280.
        self._native_dim = None
        self.proj = nn.Identity()

    def _ensure_projection(self, native_dim):
        """Build the projection on first use, once the cached width is known."""
        if self._native_dim is not None:
            return
        self._native_dim = int(native_dim)
        if self._native_dim != self.output_dim:
            self.proj = nn.Linear(self._native_dim, self.output_dim).to(self.device)

    def _load_record(self, sequence):
        key = sequence_cache_key(sequence)
        cache_path = os.path.join(self.cache_dir, f"{key}.pt")

        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"SaProtProteinEncoder: no cache entry for md5 {key} "
                f"(expected at {cache_path}; sequence length {len(sequence)}, "
                f"starts {sequence[:16]!r}). scripts/precompute_saprot.py is "
                f"supposed to cover every sequence in the split, so a miss here "
                f"is a precompute bug rather than a runtime condition."
            )

        record = torch.load(cache_path, map_location="cpu", weights_only=True)
        features = record["features"]

        if features.dim() != 2:
            raise ValueError(
                f"SaProtProteinEncoder: cache entry {key} has features of rank "
                f"{features.dim()}, expected a 2-D [L, D] tensor."
            )
        if int(record["length"]) != features.shape[0]:
            raise ValueError(
                f"SaProtProteinEncoder: cache entry {key} declares length "
                f"{int(record['length'])} but carries {features.shape[0]} rows."
            )
        return features

    def forward(self, protein_sequences):
        if not protein_sequences:
            raise ValueError("SaProtProteinEncoder: got an empty batch of sequences.")

        records = [self._load_record(sequence) for sequence in protein_sequences]
        self._ensure_projection(records[0].shape[-1])

        for index, tensor in enumerate(records):
            if tensor.shape[-1] != self._native_dim:
                raise ValueError(
                    f"SaProtProteinEncoder: cache width {tensor.shape[-1]} at batch "
                    f"index {index} differs from {self._native_dim}; the cache mixes "
                    f"records from different checkpoints."
                )

        # Project before padding: a Linear over pad rows would add its bias and
        # stop those positions being zero, diverging from the zero-fill convention
        # the ESM-2 cached path establishes (models.py:106-108).
        stacked = torch.cat(records, dim=0).to(self.device)
        projected = self.proj(stacked)
        lengths = [tensor.shape[0] for tensor in records]
        chunks = torch.split(projected, lengths, dim=0)

        max_len = max(lengths)
        features = torch.zeros(len(records), max_len, self.output_dim,
                               dtype=projected.dtype, device=self.device)
        mask = torch.zeros(len(records), max_len, dtype=torch.long, device=self.device)
        for index, (chunk, length) in enumerate(zip(chunks, lengths)):
            features[index, :length] = chunk
            mask[index, :length] = 1

        return features, mask
