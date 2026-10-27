#!/usr/bin/env python
"""Precompute heavy-atom Uni-Mol embeddings for DRUG.FUSION=unimol_gated.

One record per unique SMILES in cache/unimol_heavy/{md5(smiles)}.pt:

    {"features": FloatTensor [n, 512]   heavy atoms only, RDKit parse order
     "n_atoms": int n,
     "symbols": list[str]               element per row, parse order
     "smiles": str,
     "conformer_3d_ok": bool or None,   see CONFORMER FLAG below
     "model": "unimolv1_no_h"}

Rows stay in RDKit's SMILES-parse order here. The permutation into dgllife's
atom order (what the GCN uses) is applied at training time by
encoders/unimol_drug.py, inside the training env, so it always comes from the
same dgllife + RDKit that builds the graphs. This env has neither guarantee.

Differences from the old precompute_drug_3d.py (3d-drug-v1):
  * remove_hs=True: Uni-Mol's no-H model, so rows are heavy atoms only, the same
    atoms the GCN sees. The old cache kept hydrogens and is NOT usable here.
  * every record is checked atom-for-atom against RDKit's parse-order heavy
    atoms before it is written; a molecule that disagrees is skipped and listed
    in the report instead of being cached misaligned.
  * new cache folder (cache/unimol_heavy), so the old with-H cache can never be
    picked up by mistake.

ENVIRONMENT
Run in the Uni-Mol env (unimol_tools + RDKit), on the server:

    python scripts/precompute_unimol_heavy.py --limit 20        # trial
    python scripts/precompute_unimol_heavy.py                   # full run
    python scripts/precompute_unimol_heavy.py --self-test       # no Uni-Mol, no GPU, no data

Then, in the TRAINING env (needs dgllife), before any training run:

    python scripts/verify_unimol_alignment.py

KNOWN Uni-Mol API DETAILS (found on the 3d-drug-v1 GPU test, handled below)
  * the CPU/GPU switch is use_cuda; use_gpu is silently ignored.
  * get_repr(..., return_atomic_reprs=True) returns one extra trailing [SEP] row
    per molecule. It is dropped only after checking atomic_symbol really ends in
    '[SEP]'.

CONFORMER FLAG
Uni-Mol builds its own RDKit conformer (ETKDG, seed 42) and quietly falls back
to 2D coordinates when that fails. conformer_3d_ok re-runs the same first
attempt (AddHs + EmbedMolecule(randomSeed=42)) and records whether it worked.
It is an approximation of what Uni-Mol saw, for analysis only; skip it with
--no-conformer-check to save time.
"""

import argparse
import hashlib
import os
import sys
import time

import torch


UNIMOL_NATIVE_DIM = 512
SEP_TOKEN = "[SEP]"
MODEL_TAG = "unimolv1_no_h"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE_DIR = os.path.join(_REPO_ROOT, "cache", "unimol_heavy")


def smiles_md5(smiles):
    """Same keying as encoders/unimol_drug.py:smiles_cache_key."""
    return hashlib.md5(smiles.encode("utf-8")).hexdigest()


# --- input -----------------------------------------------------------------

def load_unique_smiles(datasets_root, dataset_names):
    """md5 -> SMILES for every unique drug in every CSV under the datasets (read-only).

    Reads every split folder, not only full.csv, so a SMILES that appears only in a
    generated cold split is still covered.
    """
    import pandas as pd

    molecules = {}
    for name in dataset_names:
        dataset_dir = os.path.join(datasets_root, name)
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")
        csv_paths = sorted(
            os.path.join(root, filename)
            for root, _, filenames in os.walk(dataset_dir)
            for filename in filenames
            if filename.endswith(".csv")
        )
        if not csv_paths:
            raise FileNotFoundError(f"no CSVs found under {dataset_dir}")
        for csv_path in csv_paths:
            frame = pd.read_csv(csv_path)
            if "SMILES" not in frame.columns:
                continue
            for smiles in frame["SMILES"].astype(str):
                molecules.setdefault(smiles_md5(smiles), smiles)
    return molecules


# --- checks ----------------------------------------------------------------

def parse_order_symbols(smiles):
    """Heavy-atom element symbols in RDKit MolFromSmiles order (what the GCN starts from)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse this SMILES")
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def conformer_3d_ok(smiles, seed=42):
    """Whether the first ETKDG attempt (Uni-Mol's own first try) gives 3D coordinates."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        return AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    except Exception:                                    # noqa: BLE001
        return False


def strip_sep_row(features, symbols):
    """Drop Uni-Mol's trailing [SEP] row, refusing to guess if it is not there."""
    if symbols is None:
        raise ValueError("Uni-Mol returned no atomic_symbol list; cannot confirm [SEP]")
    symbols = [str(s) for s in symbols]
    if len(symbols) != features.shape[0]:
        raise ValueError(f"atomic_symbol has {len(symbols)} entries but features have "
                         f"{features.shape[0]} rows")
    if symbols[-1] != SEP_TOKEN:
        raise ValueError(f"expected last atomic_symbol to be {SEP_TOKEN!r}, got "
                         f"{symbols[-1]!r}; refusing to drop a row that may be an atom")
    return features[:-1], symbols[:-1]


def build_record(smiles, features, symbols, check_conformer=True):
    """Validate one Uni-Mol output against the SMILES and return the cache record."""
    if not torch.is_tensor(features):
        features = torch.as_tensor(features)
    if features.dim() != 2 or features.shape[-1] != UNIMOL_NATIVE_DIM:
        raise ValueError(f"expected [rows, {UNIMOL_NATIVE_DIM}], got {tuple(features.shape)}")

    atoms, atom_symbols = strip_sep_row(features, symbols)
    if atoms.shape[0] == 0:
        raise ValueError("no atom rows left after dropping [SEP]")
    if "H" in atom_symbols:
        raise ValueError("hydrogen rows present; Uni-Mol was not run with remove_hs=True")

    expected = parse_order_symbols(smiles)
    if atom_symbols != expected:
        raise ValueError(f"Uni-Mol atoms {atom_symbols} != RDKit parse-order heavy atoms "
                         f"{expected}")

    return {
        "features": atoms.detach().to("cpu").float().contiguous(),
        "n_atoms": int(atoms.shape[0]),
        "symbols": atom_symbols,
        "smiles": smiles,
        "conformer_3d_ok": conformer_3d_ok(smiles) if check_conformer else None,
        "model": MODEL_TAG,
    }


def atomic_save_record(record, path):
    """Temp file + replace, so an interrupted write never leaves a truncated .pt."""
    temp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(record, temp_path)
    os.replace(temp_path, path)


# --- encoder (lazy import: absent in the self-test env) ---------------------

def build_unimol_encoder(use_cuda):
    """Return encode(list[str]) -> list of (features [n+1, 512], symbols)."""
    from unimol_tools import UniMolRepr

    model = UniMolRepr(
        data_type="molecule",
        model_name="unimolv1",
        remove_hs=True,           # heavy atoms only, same atoms as the GCN
        use_cuda=use_cuda,        # NOT use_gpu - that argument is ignored
    )

    def encode(smiles_batch):
        output = model.get_repr(smiles_batch, return_atomic_reprs=True)
        reprs = output["atomic_reprs"]
        symbols = output.get("atomic_symbol")
        if symbols is None:
            symbols = [None] * len(reprs)
        return list(zip(reprs, symbols))

    return encode


# --- precompute ------------------------------------------------------------

def run_precompute(molecules, encode, cache_dir, batch_size=16, progress_every=200,
                   limit=None, check_conformer=True):
    """Fill cache_dir from molecules (md5 -> smiles). Returns (done, already, skipped)."""
    os.makedirs(cache_dir, exist_ok=True)

    pending, already = [], 0
    for key in sorted(molecules):
        if os.path.exists(os.path.join(cache_dir, f"{key}.pt")):
            already += 1
        else:
            pending.append(key)
    if limit is not None:
        pending = pending[:limit]

    print(f"[unimol-heavy] {len(molecules)} unique SMILES, {already} already cached, "
          f"{len(pending)} to do", flush=True)

    done, skipped, processed = 0, [], 0
    for start in range(0, len(pending), batch_size):
        batch_keys = pending[start:start + batch_size]
        batch_smiles = [molecules[key] for key in batch_keys]

        try:
            outputs = encode(batch_smiles)
            if len(outputs) != len(batch_smiles):
                raise RuntimeError(f"Uni-Mol returned {len(outputs)} outputs for "
                                   f"{len(batch_smiles)} SMILES")
        except Exception:                                # noqa: BLE001
            # one bad molecule can take the whole batch down: retry singly
            outputs = []
            for smiles in batch_smiles:
                try:
                    outputs.append(encode([smiles])[0])
                except Exception as inner:               # noqa: BLE001
                    outputs.append(("__failed__", f"{type(inner).__name__}: {inner}"))

        for key, smiles, item in zip(batch_keys, batch_smiles, outputs):
            try:
                features, symbols = item
                if isinstance(features, str) and features == "__failed__":
                    raise RuntimeError(symbols)
                record = build_record(smiles, features, symbols, check_conformer)
                atomic_save_record(record, os.path.join(cache_dir, f"{key}.pt"))
                done += 1
            except Exception as exc:                     # noqa: BLE001 - reported, never fatal
                skipped.append((key, smiles, f"{type(exc).__name__}: {exc}"))

        processed += len(batch_keys)
        if processed % progress_every < batch_size or processed >= len(pending):
            print(f"  [unimol-heavy {processed}/{len(pending)}] done={done} "
                  f"skipped={len(skipped)}", flush=True)

    return done, already, skipped


# --- report ----------------------------------------------------------------

def write_report(report_path, cache_dir, molecules, done, already, skipped):
    cached = {name[:-3] for name in os.listdir(cache_dir) if name.endswith(".pt")} \
        if os.path.isdir(cache_dir) else set()
    missing = [key for key in molecules if key not in cached]

    conformer_fail = []
    for key in sorted(cached & set(molecules)):
        try:
            record = torch.load(os.path.join(cache_dir, f"{key}.pt"), map_location="cpu",
                                weights_only=True)
        except Exception:                                # noqa: BLE001
            continue
        if record.get("conformer_3d_ok") is False:
            conformer_fail.append((key, molecules[key]))

    lines = [
        "Uni-Mol heavy-atom precompute report",
        "=" * 72,
        f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"  unique SMILES expected        : {len(molecules)}",
        f"  cached records on disk        : {len(cached & set(molecules))}",
        f"  still missing                 : {len(missing)}",
        f"  written this run              : {done}",
        f"  already cached (skipped)      : {already}",
        f"  failed this run               : {len(skipped)}",
        f"  first 3D attempt failed (2D?) : {len(conformer_fail)}",
        "",
        f"FAILED - no usable record ({len(skipped)})",
        "-" * 72,
    ]
    lines += [f"  {k}  {s[:70]}\n      {r}" for k, s, r in skipped] or ["  (none)"]
    lines += ["", f"FIRST 3D ATTEMPT FAILED ({len(conformer_fail)})", "-" * 72]
    lines += [f"  {k}  {s[:70]}" for k, s in conformer_fail] or ["  (none)"]

    text = "\n".join(lines)
    temp_path = f"{report_path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    os.replace(temp_path, report_path)
    return text


# --- self-test -------------------------------------------------------------

def self_test():
    """Stubbed end-to-end check: no GPU, no unimol_tools, no real data (RDKit needed)."""
    import tempfile

    smiles_list = ["CC(=O)Oc1ccccc1C(=O)O", "OC(=O)CCN", "C1=CC=CN=C1Cl"]

    def fake_encode(batch):
        out = []
        for smiles in batch:
            symbols = parse_order_symbols(smiles) + [SEP_TOKEN]
            out.append((torch.randn(len(symbols), UNIMOL_NATIVE_DIM), symbols))
        return out

    with tempfile.TemporaryDirectory() as root:
        cache_dir = os.path.join(root, "unimol_heavy")
        molecules = {smiles_md5(s): s for s in smiles_list}
        done, already, skipped = run_precompute(molecules, fake_encode, cache_dir,
                                                batch_size=2, check_conformer=True)
        assert done == 3 and already == 0 and not skipped, (done, already, skipped)
        for key, smiles in molecules.items():
            record = torch.load(os.path.join(cache_dir, f"{key}.pt"), weights_only=True)
            assert record["n_atoms"] == len(parse_order_symbols(smiles))
            assert record["features"].shape == (record["n_atoms"], UNIMOL_NATIVE_DIM)
            assert record["symbols"] == parse_order_symbols(smiles)
            assert record["conformer_3d_ok"] in (True, False)

        # resumable: a second run writes nothing
        done2, already2, _ = run_precompute(molecules, fake_encode, cache_dir)
        assert done2 == 0 and already2 == 3

        # each malformed output must be refused, never cached
        aspirin = smiles_list[0]
        sym = parse_order_symbols(aspirin)
        bad_cases = {
            "no [SEP]": (torch.randn(len(sym), UNIMOL_NATIVE_DIM), sym),
            "hydrogens": (torch.randn(len(sym) + 2, UNIMOL_NATIVE_DIM), sym + ["H", SEP_TOKEN]),
            "wrong order": (torch.randn(len(sym) + 1, UNIMOL_NATIVE_DIM), sym[::-1] + [SEP_TOKEN]),
            "wrong width": (torch.randn(len(sym) + 1, 256), sym + [SEP_TOKEN]),
        }
        for name, (features, symbols) in bad_cases.items():
            try:
                build_record(aspirin, features, symbols, check_conformer=False)
            except ValueError:
                continue
            raise AssertionError(f"malformed case not refused: {name}")

        report = write_report(os.path.join(root, "report.txt"), cache_dir, molecules, 0, 3, [])
        assert "still missing                 : 0" in report

    print("self-test passed")
    return 0


# --- driver ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Precompute heavy-atom Uni-Mol embeddings.")
    parser.add_argument("--self-test", action="store_true",
                        help="stubbed check; no GPU, no unimol_tools, no data")
    parser.add_argument("--datasets-root", default=os.path.join(_REPO_ROOT, "datasets"))
    parser.add_argument("--datasets", nargs="+", default=["biosnap", "bindingdb"])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default=None, help="cuda or cpu; default cuda when available")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N pending molecules (trial runs)")
    parser.add_argument("--no-conformer-check", action="store_true",
                        help="skip the conformer_3d_ok flag")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    use_cuda = args.device.startswith("cuda") if args.device else torch.cuda.is_available()
    report_path = os.path.join(os.path.dirname(os.path.abspath(args.cache_dir)),
                               "unimol_heavy_report.txt")
    print(f"use_cuda: {use_cuda}\ncache: {args.cache_dir}")

    molecules = load_unique_smiles(args.datasets_root, args.datasets)
    print(f"unique SMILES across {', '.join(args.datasets)}: {len(molecules)}")

    done = already = 0
    skipped = []
    interrupted = False
    try:
        encode = build_unimol_encoder(use_cuda)
        done, already, skipped = run_precompute(
            molecules, encode, args.cache_dir, batch_size=args.batch_size,
            limit=args.limit, check_conformer=not args.no_conformer_check,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted - writing report for what completed", flush=True)

    report = write_report(report_path, args.cache_dir, molecules, done, already, skipped)
    print()
    print(report.split("FAILED")[0].rstrip())
    print(f"report -> {report_path}")
    if interrupted:
        print("re-run the same command to continue where this stopped")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
