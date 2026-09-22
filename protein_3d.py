"""Training-time 3D protein encoder.

Reads precomputed per-residue embeddings from disk and projects them into the
1280-wide space the fusion stack expects. No encoder model is loaded or run
here: the expensive ESM-IF1 / ESM-2 forward passes belong to a separate
precompute step, and this module only consumes what that step wrote.

Contract, identical to ProtBertProteinEncoder in models.py:

    forward(list[str]) -> (features [B, L_p, 1280], mask [B, L_p])

with mask 1 for a real residue and 0 for padding, and L_p the longest sequence
in the batch. The 1280 is not negotiable: the fusion attention is constructed at
input_dim = self.protein_feature_dim (models.py:265-269) and asserts that width
is divisible by BCN.HEADS = 8 (attention.py:12). 1280 / 8 = 160.
"""

import hashlib
import os

import torch
import torch.nn as nn


# Mirrors models.py:72 - the cache lives next to the source tree with one
# subdirectory per encoder. esmif1/ is deliberately kept apart from esm/: both
# key on md5(sequence) but hold different widths, so a shared directory would
# let an ESM-2 record silently satisfy a lookup meant for an ESM-IF1 one.
_CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

ESMIF1_NATIVE_DIM = 512     # ESM-IF1 per-residue width, needs projecting
ESM2_NATIVE_DIM = 1280      # ESM-2 fallback width, already the fusion width
FUSION_DIM = 1280           # what the fusion stack is sized for


def _sequence_cache_key(sequence):
    """Same keying as models.py:70-71, so precompute and training agree."""
    return hashlib.md5(sequence.encode("utf-8")).hexdigest()


class Protein3DEncoder(nn.Module):
    """Cache-backed protein encoder with a learnable projection head.

    Each cached record is one sequence:

        {"features": FloatTensor [L, D_native] (cpu),
         "length": int L,
         "has_structure": 1 if ESM-IF1 (D_native 512), 0 if ESM-2 fallback (1280)}

    The cached tensors are constants - they carry no graph back into the frozen
    encoders that produced them. The projection below is the only thing that
    learns, and it trains live with the rest of the model.
    """

    def __init__(self, cache_dir=None, device=None):
        super().__init__()
        self.cache_dir = cache_dir if cache_dir is not None else os.path.join(_CACHE_ROOT, "esmif1")
        self.device = device if device is not None else torch.device("cpu")
        self.output_dim = FUSION_DIM

        # The only learnable parameters in this module. ESM-IF1 records arrive
        # 512-wide and must be lifted to the fusion width.
        self.proj_if1 = nn.Linear(ESMIF1_NATIVE_DIM, FUSION_DIM)

        # Fallback records are already 1280-wide and pass through untouched. An
        # explicit Identity rather than an `if` at the call site, so the two
        # sources read as the same operation with different weights.
        self.proj_esm2 = nn.Identity()

        # Deliberately NOT overriding train() the way ProtBertProteinEncoder and
        # ChemBERTaEncoder do (models.py:146, :192). Those pin themselves to
        # eval() because they wrap frozen encoders whose dropout must never come
        # back on. This module holds trainable weights and must follow the parent
        # CMA module's mode like any other submodule.

    def _load_record(self, sequence):
        """Fetch one sequence's cached embedding, validating it on the way in."""
        key = _sequence_cache_key(sequence)
        cache_path = os.path.join(self.cache_dir, f"{key}.pt")

        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Protein3DEncoder: no cache entry for md5 {key} "
                f"(expected at {cache_path}; sequence length {len(sequence)}, "
                f"starts {sequence[:16]!r}). Precompute is supposed to cover every "
                f"sequence in the split, so a miss here is a precompute bug rather "
                f"than a runtime condition to work around."
            )

        record = torch.load(cache_path, map_location="cpu", weights_only=True)

        features = record["features"]
        has_structure = int(record["has_structure"])
        expected_dim = ESMIF1_NATIVE_DIM if has_structure else ESM2_NATIVE_DIM
        declared_length = int(record["length"])

        if features.dim() != 2:
            raise ValueError(
                f"Protein3DEncoder: cache entry {key} has features of rank "
                f"{features.dim()}, expected a 2-D [L, D] tensor."
            )
        if features.shape[-1] != expected_dim:
            raise ValueError(
                f"Protein3DEncoder: cache entry {key} declares has_structure="
                f"{has_structure}, which implies width {expected_dim}, but its "
                f"features are {features.shape[-1]}-wide. The record was written "
                f"with a mismatched flag."
            )
        if declared_length != features.shape[0]:
            raise ValueError(
                f"Protein3DEncoder: cache entry {key} declares length "
                f"{declared_length} but carries {features.shape[0]} residues."
            )

        return record

    def forward(self, protein_sequences):
        if not protein_sequences:
            raise ValueError("Protein3DEncoder: got an empty batch of sequences.")

        records = [self._load_record(sequence) for sequence in protein_sequences]

        # Project before padding, never after. The two sources arrive at
        # different widths so they cannot share one padded buffer, and running a
        # Linear across pad rows would add its bias to them - those positions
        # would stop being zero and diverge from the cached ESM-2 branch's
        # convention (models.py:106-108), which downstream masking assumes.
        #
        # Grouping by source keeps this to one matmul per source over the
        # concatenated residues, instead of one Linear call per sequence.
        projected = [None] * len(records)
        for has_structure, projection in ((1, self.proj_if1), (0, self.proj_esm2)):
            group = [i for i, record in enumerate(records)
                     if int(record["has_structure"]) == has_structure]
            if not group:
                continue

            stacked = torch.cat([records[i]["features"] for i in group], dim=0).to(self.device)
            group_lengths = [records[i]["features"].shape[0] for i in group]
            chunks = torch.split(projection(stacked), group_lengths, dim=0)
            for index, chunk in zip(group, chunks):
                projected[index] = chunk

        # Zero-fill padding and a 1/0 mask, matching _encode_with_cache
        # (models.py:106-115). Assigning grad-carrying rows into a plain zeros
        # buffer is differentiable, so proj_if1 still receives gradient here.
        lengths = [tensor.shape[0] for tensor in projected]
        max_len = max(lengths)
        feature_dtype = projected[0].dtype

        features = torch.zeros(len(projected), max_len, self.output_dim,
                               dtype=feature_dtype, device=self.device)
        mask = torch.zeros(len(projected), max_len, dtype=torch.long, device=self.device)
        for i, (tensor, length) in enumerate(zip(projected, lengths)):
            features[i, :length] = tensor
            mask[i, :length] = 1

        return features, mask


if __name__ == "__main__":
    import shutil
    import sys
    import tempfile

    def _write_record(cache_dir, sequence, length, native_dim, has_structure):
        """Fabricate one cache record by hand - no real model, no real data."""
        features = torch.randn(length, native_dim)
        record = {"features": features, "length": length, "has_structure": has_structure}
        torch.save(record, os.path.join(cache_dir, f"{_sequence_cache_key(sequence)}.pt"))
        return features

    cache_dir = tempfile.mkdtemp(prefix="protein3d_selftest_")
    try:
        # Two ESM-IF1 records (512-wide, different lengths) and one ESM-2
        # fallback (1280-wide), so both widths and the padding path are covered.
        seq_if1_short = "MKTAYIA"
        seq_if1_long = "MKTAYIAKQRQISFVK"
        seq_esm2 = "MKTAY"
        len_if1_short, len_if1_long, len_esm2 = 7, 16, 5
        max_len = len_if1_long

        _write_record(cache_dir, seq_if1_short, len_if1_short, ESMIF1_NATIVE_DIM, 1)
        _write_record(cache_dir, seq_if1_long, len_if1_long, ESMIF1_NATIVE_DIM, 1)
        esm2_features = _write_record(cache_dir, seq_esm2, len_esm2, ESM2_NATIVE_DIM, 0)

        encoder = Protein3DEncoder(cache_dir=cache_dir, device=torch.device("cpu"))
        features, mask = encoder([seq_if1_short, seq_if1_long, seq_esm2])

        # --- shapes and dtypes match the ESM-2 encoder's contract
        assert features.shape == (3, max_len, FUSION_DIM), \
            f"features shape {tuple(features.shape)} != (3, {max_len}, {FUSION_DIM})"
        assert mask.shape == (3, max_len), f"mask shape {tuple(mask.shape)} != (3, {max_len})"
        assert mask.dtype == torch.long, f"mask dtype {mask.dtype} != torch.long"
        assert features.dtype == torch.float32, f"features dtype {features.dtype} != torch.float32"

        # --- mask marks real residues, padding is zero-filled
        for i, length in enumerate([len_if1_short, len_if1_long, len_esm2]):
            assert int(mask[i, :length].sum()) == length, f"row {i}: real positions are not all 1"
            assert int(mask[i, length:].sum()) == 0, f"row {i}: pad positions are not all 0"
            assert torch.count_nonzero(features[i, length:]) == 0, \
                f"row {i}: padding is not zero-filled"

        # --- both source widths produce usable 1280 output
        assert torch.count_nonzero(features[0, :len_if1_short]) > 0, \
            "512-wide ESM-IF1 record projected to all zeros"
        assert torch.allclose(features[2, :len_esm2], esm2_features), \
            "1280-wide ESM-2 fallback rows were altered instead of passed through"

        # --- proj_if1 is live, so it will actually train
        assert features.requires_grad, "features do not require grad; proj_if1 would never train"
        features.sum().backward()
        assert encoder.proj_if1.weight.grad is not None, "no gradient reached proj_if1.weight"
        assert torch.count_nonzero(encoder.proj_if1.weight.grad) > 0, \
            "proj_if1.weight gradient is all zeros"

        # --- a cache miss names the md5 rather than failing quietly
        missing = "QQQNOTINCACHE"
        try:
            encoder([missing])
        except FileNotFoundError as exc:
            assert _sequence_cache_key(missing) in str(exc), \
                "cache-miss error does not name the offending md5"
        else:
            raise AssertionError("a cache miss did not raise")

        print("SELF-TEST PASS")
    except Exception as exc:
        print(f"SELF-TEST FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
