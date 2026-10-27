#!/usr/bin/env python
"""Check the Uni-Mol heavy-atom cache lines up with the GCN graphs, atom for atom.

Run in the TRAINING env (needs dgllife + RDKit, no GPU), after
scripts/precompute_unimol_heavy.py and before any DRUG.FUSION=unimol_gated run:

    python scripts/verify_unimol_alignment.py
    python scripts/verify_unimol_alignment.py --self-test     # fake cache, no data

For every unique SMILES in the datasets it:
  1. loads the record and permutes it into dgllife order exactly as training does
     (encoders/unimol_drug.py:load_aligned_record - element check + permutation),
  2. rebuilds the graph the way dataloader.py builds it (smiles_to_bigraph,
     add_self_loop=True, CanonicalAtomFeaturizer) and checks the atom featurizer
     output of the permuted molecule equals that graph's node features exactly,
  3. checks the atom count fits DRUG.MAX_NODES.

Exit code 1 if any record is misaligned. Missing records are reported but do not
fail the check (training with DRUG.UNIMOL_MISSING=error will stop on them).
"""

import argparse
import os
import sys
import time
from functools import partial

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from encoders.unimol_drug import (  # noqa: E402
    DEFAULT_CACHE_DIR, dgllife_atom_order, load_aligned_record, smiles_cache_key,
)


def dataloader_graph_features(smiles):
    """Node features of the graph dataloader.py builds, before its padding step."""
    from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

    build = partial(smiles_to_bigraph, add_self_loop=True)
    graph = build(smiles=smiles, node_featurizer=CanonicalAtomFeaturizer(),
                  edge_featurizer=CanonicalBondFeaturizer(self_loop=True))
    return graph.ndata["h"]


def check_one(smiles, cache_dir, max_nodes):
    """Return (status, detail). status in ok / missing / mismatch / too_many_atoms."""
    try:
        aligned = load_aligned_record(smiles, cache_dir)
    except FileNotFoundError:
        return "missing", ""
    except Exception as exc:                             # noqa: BLE001
        return "mismatch", f"{type(exc).__name__}: {exc}"

    perm, our_features = dgllife_atom_order(smiles, with_features=True)
    graph_features = dataloader_graph_features(smiles)
    if our_features.shape != graph_features.shape or not torch.equal(our_features, graph_features):
        return "mismatch", "atom features of the permuted molecule differ from the dataloader graph"
    if aligned.shape[0] != graph_features.shape[0]:
        return "mismatch", f"{aligned.shape[0]} Uni-Mol rows vs {graph_features.shape[0]} graph nodes"
    if aligned.shape[0] > max_nodes:
        return "too_many_atoms", f"{aligned.shape[0]} > {max_nodes}"
    return "ok", ""


def load_smiles(datasets_root, dataset_names):
    sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
    from precompute_unimol_heavy import load_unique_smiles
    return load_unique_smiles(datasets_root, dataset_names)


def run(molecules, cache_dir, max_nodes):
    buckets = {"ok": [], "missing": [], "mismatch": [], "too_many_atoms": []}
    for count, (key, smiles) in enumerate(sorted(molecules.items()), 1):
        status, detail = check_one(smiles, cache_dir, max_nodes)
        buckets[status].append((key, smiles, detail))
        if count % 1000 == 0:
            print(f"  checked {count}/{len(molecules)}", flush=True)
    return buckets


def format_report(buckets, total):
    lines = ["Uni-Mol / GCN alignment check", "=" * 72,
             f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"  unique SMILES      : {total}"]
    lines += [f"  {name:<19}: {len(items)}" for name, items in buckets.items()]
    for name in ("mismatch", "too_many_atoms", "missing"):
        lines += ["", f"{name.upper()} ({len(buckets[name])})", "-" * 72]
        lines += [f"  {k}  {s[:70]}" + (f"\n      {d}" if d else "")
                  for k, s, d in buckets[name]] or ["  (none)"]
    return "\n".join(lines)


def self_test():
    """Fake cache in a temp dir: aligned records pass, a mis-ordered record is caught."""
    import tempfile
    sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
    from precompute_unimol_heavy import (SEP_TOKEN, UNIMOL_NATIVE_DIM, atomic_save_record,
                                         build_record, parse_order_symbols)

    smiles_list = ["CC(=O)Oc1ccccc1C(=O)O", "OC(=O)CCN", "C1=CC=CN=C1Cl", "c1ccc2ccccc2c1"]
    with tempfile.TemporaryDirectory() as cache_dir:
        for smiles in smiles_list:
            symbols = parse_order_symbols(smiles) + [SEP_TOKEN]
            record = build_record(smiles, torch.randn(len(symbols), UNIMOL_NATIVE_DIM),
                                  symbols, check_conformer=False)
            atomic_save_record(record, os.path.join(cache_dir, f"{smiles_cache_key(smiles)}.pt"))

        molecules = {smiles_cache_key(s): s for s in smiles_list + ["CCO"]}   # CCO never cached
        buckets = run(molecules, cache_dir, max_nodes=290)
        assert len(buckets["ok"]) == 4 and len(buckets["missing"]) == 1, \
            {k: len(v) for k, v in buckets.items()}

        # corrupt one record's atom order -> must be caught
        smiles = smiles_list[2]
        path = os.path.join(cache_dir, f"{smiles_cache_key(smiles)}.pt")
        record = torch.load(path, weights_only=True)
        record["symbols"] = record["symbols"][::-1]
        torch.save(record, path)
        status, _ = check_one(smiles, cache_dir, 290)
        assert status == "mismatch", status

        # too many atoms for MAX_NODES -> caught
        status, _ = check_one(smiles_list[0], cache_dir, 5)
        assert status == "too_many_atoms", status

    print("self-test passed")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Verify Uni-Mol cache vs GCN atom order.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--datasets-root", default=os.path.join(_REPO_ROOT, "datasets"))
    parser.add_argument("--datasets", nargs="+", default=["biosnap", "bindingdb"])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-nodes", type=int, default=290)
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    molecules = load_smiles(args.datasets_root, args.datasets)
    print(f"unique SMILES across {', '.join(args.datasets)}: {len(molecules)}")
    buckets = run(molecules, args.cache_dir, args.max_nodes)
    report = format_report(buckets, len(molecules))
    report_path = os.path.join(os.path.dirname(os.path.abspath(args.cache_dir)),
                               "unimol_heavy_alignment.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report + "\n")
    print(report.split("MISMATCH")[0].rstrip())
    print(f"report -> {report_path}")
    return 1 if (buckets["mismatch"] or buckets["too_many_atoms"]) else 0


if __name__ == "__main__":
    sys.exit(main())
