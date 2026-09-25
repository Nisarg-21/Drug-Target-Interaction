"""Training-time 3D drug encoder.

Reads precomputed per-atom Uni-Mol embeddings from disk and hands them to the
model padded to the fixed node width the fusion stack expects. No encoder model
is loaded or run here: the Uni-Mol forward passes happen in a separate
precompute step, and this module only consumes what that step wrote.

Contract, the drug-side counterpart of protein_3d.py's Protein3DEncoder:

    forward(list[str]) -> (features [B, 290, 512], mask [B, 290])

with mask 1 for a real atom and 0 for padding.

Two things differ deliberately from the protein encoder:

  * The node axis is a FIXED 290, not the batch maximum. DRUG.MAX_NODES is a
    hard cap throughout the model - the dataloader pads every molecule to it
    before batching and MolecularGCN raises outright if a batch does not match
    (models.py:371-375) - so this encoder pads and truncates to the same width
    rather than to whatever the batch happens to contain.

  * There is NO projection, and no learnable parameters at all. The model
    already owns gcn_proj_for_cross_attn (models.py:256), a Linear that adapts
    the drug width into the cross-attention, exactly as it did for the GCN's
    raw output. This encoder therefore emits Uni-Mol's native 512 and leaves
    the adapting to the layer that already does it.
"""

import hashlib
import os

import torch
import torch.nn as nn


# Mirrors protein_3d.py:28 - one cache subdirectory per encoder. unimol/ is kept
# apart from esm/ and esmif1/: all three key on an md5 of their input string, so
# sharing a directory would let one encoder's record satisfy another's lookup.
_CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

UNIMOL_NATIVE_DIM = 512     # Uni-Mol per-atom width, passed through unchanged
DEFAULT_MAX_NODES = 290     # DRUG.MAX_NODES (configs.py:12)


def _smiles_cache_key(smiles):
    """md5 of the raw SMILES string, matching the precompute step's keying."""
    return hashlib.md5(smiles.encode("utf-8")).hexdigest()


class Drug3DEncoder(nn.Module):
    """Cache-backed per-atom drug encoder.

    Each cached record is one molecule:

        {"features": FloatTensor [n_atoms, 512] (cpu), "n_atoms": int}

    n_atoms is the real atom count. Uni-Mol's trailing [SEP] row is dropped at
    precompute time, so the feature rows are atoms and nothing else - there is
    no boundary token to strip here.

    Holds no parameters: the cached tensors are constants and nothing in this
    module learns. It is still an nn.Module so it can sit in the model's
    submodule tree like the encoder it replaces.
    """

    def __init__(self, cache_dir=None, device=None, max_nodes=DEFAULT_MAX_NODES):
        super().__init__()
        self.cache_dir = cache_dir if cache_dir is not None else os.path.join(_CACHE_ROOT, "unimol")
        self.device = device if device is not None else torch.device("cpu")
        self.max_nodes = int(max_nodes)
        self.output_dim = UNIMOL_NATIVE_DIM

    def _load_record(self, smiles):
        """Fetch one molecule's cached embedding, validating it on the way in."""
        key = _smiles_cache_key(smiles)
        cache_path = os.path.join(self.cache_dir, f"{key}.pt")

        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Drug3DEncoder: no cache entry for md5 {key} "
                f"(expected at {cache_path}; SMILES length {len(smiles)}, "
                f"starts {smiles[:32]!r}). Precompute is supposed to cover every "
                f"molecule in the split, so a miss here is a precompute bug rather "
                f"than a runtime condition to work around."
            )

        record = torch.load(cache_path, map_location="cpu", weights_only=True)

        features = record["features"]
        n_atoms = int(record["n_atoms"])

        if features.dim() != 2:
            raise ValueError(
                f"Drug3DEncoder: cache entry {key} has features of rank "
                f"{features.dim()}, expected a 2-D [n_atoms, D] tensor."
            )
        if features.shape[-1] != UNIMOL_NATIVE_DIM:
            raise ValueError(
                f"Drug3DEncoder: cache entry {key} is {features.shape[-1]}-wide, "
                f"expected Uni-Mol's native {UNIMOL_NATIVE_DIM}."
            )
        if n_atoms != features.shape[0]:
            raise ValueError(
                f"Drug3DEncoder: cache entry {key} declares n_atoms {n_atoms} "
                f"but carries {features.shape[0]} rows. The [SEP] row should "
                f"already have been dropped at precompute time."
            )

        return features

    def forward(self, smiles_list):
        if not smiles_list:
            raise ValueError("Drug3DEncoder: got an empty batch of SMILES.")

        molecules = [self._load_record(smiles) for smiles in smiles_list]

        batch_size = len(molecules)
        feature_dtype = molecules[0].dtype

        # Zero-filled padding plus a 1/0 mask, the same convention protein_3d.py
        # uses and the same one the DGL ndata['node_mask'] carried on the drug
        # side (dataloader.py:51-54, models.py:292): 1 marks a real atom.
        features = torch.zeros(batch_size, self.max_nodes, self.output_dim,
                               dtype=feature_dtype, device=self.device)
        mask = torch.zeros(batch_size, self.max_nodes, dtype=torch.long,
                           device=self.device)

        for index, molecule in enumerate(molecules):
            # Molecules above the cap are truncated rather than rejected: 290 is
            # the model's hard node limit, and the old GCN path silently dropped
            # the overflow too by never giving it a row.
            kept = min(molecule.shape[0], self.max_nodes)
            features[index, :kept] = molecule[:kept].to(device=self.device,
                                                        dtype=feature_dtype)
            mask[index, :kept] = 1

        return features, mask


if __name__ == "__main__":
    import shutil
    import sys
    import tempfile

    def _write_record(cache_dir, smiles, n_atoms):
        """Fabricate one cache record by hand - no real model, no real data."""
        features = torch.randn(n_atoms, UNIMOL_NATIVE_DIM)
        record = {"features": features, "n_atoms": n_atoms}
        torch.save(record, os.path.join(cache_dir, f"{_smiles_cache_key(smiles)}.pt"))
        return features

    cache_dir = tempfile.mkdtemp(prefix="drug3d_selftest_")
    try:
        # Three molecules: two comfortably under the cap, one above it so the
        # truncation path is exercised.
        smiles_small = "CCO"
        smiles_mid = "CC(=O)Oc1ccccc1C(=O)O"
        smiles_big = "C" * 60
        n_small, n_mid, n_big = 12, 40, 305

        features_small = _write_record(cache_dir, smiles_small, n_small)
        _write_record(cache_dir, smiles_mid, n_mid)
        features_big = _write_record(cache_dir, smiles_big, n_big)

        encoder = Drug3DEncoder(cache_dir=cache_dir, device=torch.device("cpu"))
        features, mask = encoder([smiles_small, smiles_mid, smiles_big])

        max_nodes = DEFAULT_MAX_NODES

        # --- shapes and dtypes
        assert features.shape == (3, max_nodes, UNIMOL_NATIVE_DIM), \
            f"features shape {tuple(features.shape)} != (3, {max_nodes}, {UNIMOL_NATIVE_DIM})"
        assert mask.shape == (3, max_nodes), f"mask shape {tuple(mask.shape)} != (3, {max_nodes})"
        assert mask.dtype == torch.long, f"mask dtype {mask.dtype} != torch.long"
        assert encoder.output_dim == UNIMOL_NATIVE_DIM, "output_dim must be 512"

        # --- under-cap molecules: exactly n_atoms ones, rest zero, padding zeroed
        for row, n_atoms in ((0, n_small), (1, n_mid)):
            assert int(mask[row].sum()) == n_atoms, \
                f"row {row}: mask has {int(mask[row].sum())} ones, expected {n_atoms}"
            assert int(mask[row, :n_atoms].sum()) == n_atoms, f"row {row}: real atoms not all 1"
            assert int(mask[row, n_atoms:].sum()) == 0, f"row {row}: pad positions not all 0"
            assert torch.count_nonzero(features[row, n_atoms:]) == 0, \
                f"row {row}: padding is not zero-filled"

        # --- real rows carry the cached values through unchanged (no projection)
        assert torch.allclose(features[0, :n_small], features_small), \
            "real atom rows were altered instead of passed through"

        # --- over-cap molecule: truncated to exactly max_nodes, mask all ones
        assert int(mask[2].sum()) == max_nodes, \
            f"over-cap row: mask sum {int(mask[2].sum())} != {max_nodes}"
        assert torch.allclose(features[2], features_big[:max_nodes]), \
            "over-cap molecule was not truncated to the first max_nodes atoms"

        # --- no learnable parameters
        assert len(list(encoder.parameters())) == 0, \
            "Drug3DEncoder must hold no parameters"
        assert not features.requires_grad, "cached features should not carry grad"

        # --- a cache miss names the md5 rather than failing quietly
        missing = "N#Cc1ccccc1NOTINCACHE"
        try:
            encoder([missing])
        except FileNotFoundError as exc:
            assert _smiles_cache_key(missing) in str(exc), \
                "cache-miss error does not name the offending md5"
        else:
            raise AssertionError("a cache miss did not raise")

        print("SELF-TEST PASS")
    except Exception as exc:
        print(f"SELF-TEST FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
