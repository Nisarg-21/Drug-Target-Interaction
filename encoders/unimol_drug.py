"""Uni-Mol drug branch for approach 1: GCN + Uni-Mol, merged per atom by a scalar gate.

Three pieces, all used only when DRUG.FUSION == "unimol_gated" (models.py):

  dgllife_atom_order     the atom permutation dgllife applies inside
                         smiles_to_bigraph. Uni-Mol rows come out in RDKit's
                         SMILES-parse order; the GCN rows are in dgllife's
                         canonical order (canonical_atom_order=True by default,
                         which RenumberAtoms the molecule). The two are NOT the
                         same order, so every Uni-Mol record is permuted here
                         before it is put next to the GCN output.

  UniMolDrugEncoder      training-time, cache-backed, no parameters. Reads the
                         heavy-atom records scripts/precompute_unimol_heavy.py
                         wrote, permutes them into dgllife order and pads to
                         DRUG.MAX_NODES, exactly the node layout the GCN uses.

  GatedAtomFusion        the trainable part. For atom i:

                             U_i = LN(ReLU(W_a LN(u_i)))          512 -> 1280
                             z_i = sigmoid(w_z . [G_i || U_i] + b) 2560 -> 1
                             D_i = z_i * G_i + (1 - z_i) * U_i     1280

                         G_i is the (already LayerNorm-ed) GCN vector, u_i the
                         frozen Uni-Mol vector. z_i is ONE number per atom: the
                         share given to the GCN view. w_z and b start at zero,
                         so every atom starts at z = 0.5.

The record format (one file per unique SMILES, cache/unimol_heavy/{md5}.pt):

    {"features": FloatTensor [n, 512]   rows in RDKit parse order, heavy atoms only
     "n_atoms": int n,
     "symbols": list[str] (len n)       element of each row, parse order
     "smiles": str,
     "conformer_3d_ok": bool or None,
     "model": "unimolv1_no_h"}

Why the permutation is computed here (training env) and not at precompute time:
the precompute runs in the separate Uni-Mol env, which has no dgllife and may
have a different RDKit. Canonical atom ranking can change between RDKit
versions, so the permutation has to come from the same dgllife + RDKit that
builds the training graphs. Asking dgllife itself for it (below) also means it
stays correct whatever its defaults are.
"""

import hashlib
import inspect
import os

import torch
import torch.nn as nn


UNIMOL_NATIVE_DIM = 512
_CACHE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
DEFAULT_CACHE_DIR = os.path.join(_CACHE_ROOT, "unimol_heavy")
_ORIG_IDX_PROP = "_cmadti_orig_idx"


def smiles_cache_key(smiles):
    """md5 of the raw SMILES string; same keying as every other cache in the repo."""
    return hashlib.md5(smiles.encode("utf-8")).hexdigest()


def parse_order_symbols(smiles):
    """Element symbols of RDKit's MolFromSmiles atoms, in parse order (no added H)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def dgllife_atom_order(smiles, with_features=False):
    """Return perm such that dgllife graph node k is parse-order atom perm[k].

    Built by asking dgllife directly: tag every atom with its parse index, then
    run the same mol_to_bigraph call smiles_to_bigraph makes (same defaults, read
    off smiles_to_bigraph's own signature) with a featurizer that reads the tags
    back off the molecule dgllife actually featurizes. RenumberAtoms keeps atom
    properties, so the tags travel with the atoms.

    with_features=True also returns the CanonicalAtomFeaturizer matrix of that
    molecule, so scripts/verify_unimol_alignment.py can compare it to the graph
    the real dataloader builds.
    """
    from rdkit import Chem
    from dgllife.utils import mol_to_bigraph, smiles_to_bigraph, CanonicalAtomFeaturizer

    defaults = inspect.signature(smiles_to_bigraph).parameters
    canonical_atom_order = defaults["canonical_atom_order"].default
    explicit_hydrogens = defaults["explicit_hydrogens"].default

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetIntProp(_ORIG_IDX_PROP, atom.GetIdx())

    atom_featurizer = CanonicalAtomFeaturizer() if with_features else None

    def featurize(m):
        out = {"orig_idx": torch.tensor([a.GetIntProp(_ORIG_IDX_PROP) for a in m.GetAtoms()],
                                        dtype=torch.long)}
        if atom_featurizer is not None:
            out["h"] = atom_featurizer(m)["h"]
        return out

    graph = mol_to_bigraph(mol, node_featurizer=featurize,
                           canonical_atom_order=canonical_atom_order,
                           explicit_hydrogens=explicit_hydrogens)
    perm = graph.ndata["orig_idx"].tolist()
    if with_features:
        return perm, graph.ndata["h"]
    return perm


def load_aligned_record(smiles, cache_dir):
    """Load one cached record and return its features permuted into dgllife order.

    Raises FileNotFoundError if the SMILES was never precomputed, ValueError if
    the record does not match this SMILES atom for atom.
    """
    key = smiles_cache_key(smiles)
    path = os.path.join(cache_dir, f"{key}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no Uni-Mol cache entry for {smiles!r} (expected {path})")

    record = torch.load(path, map_location="cpu", weights_only=True)
    features = record["features"]
    n_atoms = int(record["n_atoms"])
    symbols = list(record["symbols"])

    if features.dim() != 2 or features.shape[-1] != UNIMOL_NATIVE_DIM:
        raise ValueError(f"{key}: features shape {tuple(features.shape)}, "
                         f"expected [n, {UNIMOL_NATIVE_DIM}]")
    if features.shape[0] != n_atoms or len(symbols) != n_atoms:
        raise ValueError(f"{key}: n_atoms {n_atoms}, rows {features.shape[0]}, "
                         f"symbols {len(symbols)} disagree")

    expected = parse_order_symbols(smiles)
    if symbols != expected:
        raise ValueError(f"{key}: cached atoms {symbols} do not match RDKit parse order "
                         f"{expected} for {smiles!r}")

    perm = dgllife_atom_order(smiles)
    if sorted(perm) != list(range(n_atoms)):
        raise ValueError(f"{key}: dgllife graph has {len(perm)} atoms, cache has {n_atoms}")

    return features[perm].float().contiguous()


class UniMolDrugEncoder(nn.Module):
    """Cache-backed per-atom Uni-Mol features in the GCN's node layout.

        forward(list[str]) -> (features [B, max_nodes, 512], n_atoms [B])

    features rows follow dgllife's atom order and are zero past the real atoms.
    n_atoms is the real atom count, or -1 for a molecule with no cache entry
    when missing == "zeros" (its rows are all zero, so the gate can lean on the
    GCN for it). With missing == "error" (default) a miss stops the run.

    No parameters. Aligned records are kept in memory after first use (one
    dataset's unique drugs is well under 1 GB at float32), which saves re-reading
    and re-permuting the same molecule every batch; results are identical either way.
    """

    def __init__(self, cache_dir=None, max_nodes=290, missing="error"):
        super().__init__()
        if missing not in ("error", "zeros"):
            raise ValueError(f"missing must be 'error' or 'zeros', got {missing!r}")
        self.cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
        self.max_nodes = int(max_nodes)
        self.missing = missing
        self.output_dim = UNIMOL_NATIVE_DIM
        self._memo = {}

    def train(self, mode=True):
        # no parameters and no dropout; pinned like the other frozen encoders
        return super().train(False)

    def _aligned(self, smiles):
        if smiles in self._memo:
            return self._memo[smiles]
        try:
            features = load_aligned_record(smiles, self.cache_dir)
        except FileNotFoundError:
            if self.missing == "error":
                raise
            features = None
        if features is not None and features.shape[0] > self.max_nodes:
            raise ValueError(f"{smiles!r} has {features.shape[0]} atoms, more than "
                             f"DRUG.MAX_NODES={self.max_nodes}")
        self._memo[smiles] = features
        return features

    def forward(self, smiles_sequences):
        batch = len(smiles_sequences)
        features = torch.zeros(batch, self.max_nodes, UNIMOL_NATIVE_DIM)
        n_atoms = torch.full((batch,), -1, dtype=torch.long)
        for row, smiles in enumerate(smiles_sequences):
            aligned = self._aligned(smiles)
            if aligned is None:
                continue
            features[row, :aligned.shape[0]] = aligned
            n_atoms[row] = aligned.shape[0]
        return features, n_atoms


class GatedAtomFusion(nn.Module):
    """Scalar per-atom gate between the GCN view and the Uni-Mol view.

    forward(g, u) -> (d, z)
        g  [B, N, gcn_dim]       LayerNorm-ed GCN output (models.py does the LN,
                                 shared with the gcn_only baseline)
        u  [B, N, unimol_dim]    raw frozen Uni-Mol features (zeros on padding)
        d  [B, N, gcn_dim]       fused atom vectors, go straight to the protein BCN
        z  [B, N, 1]             GCN share per atom, in (0, 1)
    """

    def __init__(self, gcn_dim=1280, unimol_dim=UNIMOL_NATIVE_DIM):
        super().__init__()
        self.unimol_in_norm = nn.LayerNorm(unimol_dim)
        self.unimol_adapter = nn.Linear(unimol_dim, gcn_dim)      # trainable Uni-Mol path
        self.unimol_out_norm = nn.LayerNorm(gcn_dim)
        self.gate = nn.Linear(2 * gcn_dim, 1)                     # 2560 weights + 1 bias
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)                            # z = 0.5 for every atom at start

    def forward(self, g, u):
        u_proj = self.unimol_out_norm(torch.relu(self.unimol_adapter(self.unimol_in_norm(u))))
        z = torch.sigmoid(self.gate(torch.cat([g, u_proj], dim=-1)))
        d = z * g + (1.0 - z) * u_proj
        return d, z
