#!/usr/bin/env python
"""Precompute per-atom Uni-Mol embeddings into the cache drug_3d.py reads.

Writes one record per molecule to cache/unimol/{md5(smiles)}.pt in exactly the
format Drug3DEncoder._load_record validates (drug_3d.py:75-110):

    {"features": FloatTensor [n_atoms, 512] (cpu), "n_atoms": int}

n_atoms is the real atom count. Uni-Mol emits a trailing [SEP] row alongside the
atom rows; it is dropped here, so the cached rows are atoms and nothing else and
drug_3d.py has no boundary token to strip at training time.

ENVIRONMENT
Runs under ~/test_3d_env_unimol (unimol_tools + its own torch/numpy pin), which
is deliberately separate from the training env for the same reason the protein
precompute is split: the stacks pin incompatible numpy versions.

    python precompute_drug_3d.py                       # full run, gpu
    python precompute_drug_3d.py --limit 5             # trial
    python precompute_drug_3d.py --device cpu          # force cpu
    python precompute_drug_3d.py --self-test           # no gpu, no unimol, no data

TWO THINGS THAT BIT US ON THE GPU TEST, BOTH HANDLED BELOW
  * The CPU/GPU switch is use_cuda. Passing use_gpu=False does NOT keep the
    model on the CPU - it is silently ignored and the run lands on the GPU.
  * get_repr() returns one extra row per molecule: a trailing [SEP]. Dropping it
    is not optional - keeping it would shift every atom index by one against the
    mask drug_3d.py builds. The atomic_symbol list is checked to actually end in
    '[SEP]' before the row is dropped, so a future API change surfaces as a
    skipped molecule in the report rather than a silently misaligned record.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch


UNIMOL_NATIVE_DIM = 512
SEP_TOKEN = "[SEP]"

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def smiles_md5(smiles):
    """Same keying as drug_3d.py:44."""
    return hashlib.md5(smiles.encode("utf-8")).hexdigest()


# --- input -----------------------------------------------------------------

def load_unique_smiles(datasets_root, dataset_names):
    """md5 -> SMILES for every unique drug in the datasets (read-only)."""
    import pandas as pd

    molecules = {}
    for name in dataset_names:
        dataset_dir = os.path.join(datasets_root, name)
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")

        full_csv = os.path.join(dataset_dir, "full.csv")
        if os.path.exists(full_csv):
            csv_paths = [full_csv]
        else:
            csv_paths = sorted(
                os.path.join(root, filename)
                for root, _, filenames in os.walk(dataset_dir)
                for filename in filenames
                if filename.endswith(".csv")
            )
            if not csv_paths:
                raise FileNotFoundError(f"no CSVs found under {dataset_dir}")

        for csv_path in csv_paths:
            frame = pd.read_csv(csv_path, usecols=["SMILES"])
            for smiles in frame["SMILES"].astype(str):
                molecules.setdefault(smiles_md5(smiles), smiles)

    return molecules


# --- atomic writes ---------------------------------------------------------

def atomic_save_record(record, path):
    """Temp file + replace, so an interrupted write cannot leave a truncated .pt
    that a resumed run would skip as already done."""
    temp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(record, temp_path)
    os.replace(temp_path, path)


def make_record(features):
    """Build a record in exactly drug_3d.py's expected shape."""
    if not torch.is_tensor(features):
        features = torch.as_tensor(features)
    features = features.detach().to("cpu").float().contiguous()
    return {"features": features, "n_atoms": int(features.shape[0])}


def strip_sep_row(features, symbols):
    """Drop Uni-Mol's trailing [SEP] row, refusing to guess if it is not there.

    Returns the atom rows only. Raises if the symbol list does not end in
    [SEP] or if the two disagree on length - either means the layout changed
    and dropping the last row blindly would silently misalign every atom.
    """
    if symbols is None:
        raise ValueError("Uni-Mol returned no atomic_symbol list; cannot confirm [SEP]")
    if len(symbols) != features.shape[0]:
        raise ValueError(
            f"atomic_symbol has {len(symbols)} entries but features have "
            f"{features.shape[0]} rows"
        )
    if str(symbols[-1]) != SEP_TOKEN:
        raise ValueError(
            f"expected the last atomic_symbol to be {SEP_TOKEN!r}, got "
            f"{str(symbols[-1])!r}; refusing to drop a row that may be an atom"
        )
    return features[:-1]


# --- encoder (lazy import: absent in the self-test env) ---------------------

def build_unimol_encoder(use_cuda):
    """Return encode(list[str]) -> list of (features [n+1, 512], symbols).

    Imported here rather than at module scope so --self-test runs in an
    environment without unimol_tools installed.

    remove_hs=False keeps hydrogens, and UniMolRepr generates the RDKit
    conformer from the SMILES itself, so no explicit RDKit step is needed here.
    """
    from unimol_tools import UniMolRepr

    model = UniMolRepr(
        data_type="molecule",
        model_name="unimolv1",
        remove_hs=False,          # with hydrogens
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

def run_precompute(molecules, encode, cache_dir, batch_size=16,
                   progress_every=200, limit=None):
    """Fill cache_dir from molecules (md5 -> smiles). Returns (done, already, skipped)."""
    os.makedirs(cache_dir, exist_ok=True)

    pending = []
    already = 0
    for key in sorted(molecules):
        if os.path.exists(os.path.join(cache_dir, f"{key}.pt")):
            already += 1
            continue
        pending.append(key)
    if limit is not None:
        pending = pending[:limit]

    print(f"[unimol] {len(molecules)} unique SMILES, {already} already cached, "
          f"{len(pending)} to do", flush=True)

    done = 0
    skipped = []
    processed = 0

    for start in range(0, len(pending), batch_size):
        batch_keys = pending[start:start + batch_size]
        batch_smiles = [molecules[key] for key in batch_keys]

        try:
            outputs = encode(batch_smiles)
        except Exception as exc:                       # noqa: BLE001
            # One unparseable molecule can take the whole batch down, so retry
            # singly to isolate it rather than losing the other 15.
            outputs = []
            for key, smiles in zip(batch_keys, batch_smiles):
                try:
                    outputs.append(encode([smiles])[0])
                except Exception as inner:             # noqa: BLE001
                    outputs.append(("__failed__", f"{type(inner).__name__}: {inner}"))
            del exc

        for key, item in zip(batch_keys, outputs):
            try:
                features, symbols = item
                if isinstance(features, str) and features == "__failed__":
                    raise RuntimeError(symbols)

                if not torch.is_tensor(features):
                    features = torch.as_tensor(features)
                if features.dim() != 2:
                    raise ValueError(f"expected a 2-D [rows, D] tensor, got "
                                     f"{tuple(features.shape)}")
                if features.shape[-1] != UNIMOL_NATIVE_DIM:
                    raise ValueError(f"expected width {UNIMOL_NATIVE_DIM}, got "
                                     f"{features.shape[-1]}")

                atoms_only = strip_sep_row(features, symbols)
                if atoms_only.shape[0] == 0:
                    raise ValueError("no atom rows left after dropping [SEP]")

                atomic_save_record(make_record(atoms_only),
                                   os.path.join(cache_dir, f"{key}.pt"))
                done += 1
            except Exception as exc:                   # noqa: BLE001 - reported, never fatal
                skipped.append((key, molecules[key], f"{type(exc).__name__}: {exc}"))

        processed += len(batch_keys)
        if processed % progress_every < batch_size or processed >= len(pending):
            print(f"  [unimol {processed}/{len(pending)}] done={done} "
                  f"skipped={len(skipped)}", flush=True)

    return done, already, skipped


# --- report ----------------------------------------------------------------

def write_report(report_path, cache_dir, expected_total, done, already, skipped):
    cached = {
        name[:-3] for name in os.listdir(cache_dir) if name.endswith(".pt")
    } if os.path.isdir(cache_dir) else set()

    lines = []
    lines.append("Uni-Mol drug precompute report")
    lines.append("=" * 72)
    lines.append(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"  unique SMILES expected    : {expected_total}")
    lines.append(f"  cached records on disk    : {len(cached)}")
    lines.append(f"  still missing             : {expected_total - len(cached)}")
    lines.append("")
    lines.append(f"  written this run          : {done}")
    lines.append(f"  already cached (skipped)  : {already}")
    lines.append(f"  failed / skipped          : {len(skipped)}")
    lines.append("")

    lines.append(f"SKIPPED - Uni-Mol could not produce a usable record ({len(skipped)})")
    lines.append("-" * 72)
    if not skipped:
        lines.append("  (none)")
    for key, smiles, reason in skipped:
        lines.append(f"  {key}  {smiles[:60]}")
        lines.append(f"      {reason}")
    lines.append("")

    text = "\n".join(lines)
    temp_path = f"{report_path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    os.replace(temp_path, report_path)
    return text


# --- self-test -------------------------------------------------------------

def self_test():
    """Stubbed end-to-end check: no gpu, no unimol_tools, no real data."""
    import shutil
    import tempfile

    root = tempfile.mkdtemp(prefix="precompute_drug_selftest_")
    try:
        cache_dir = os.path.join(root, "cache", "unimol")
        report_path = os.path.join(root, "cache", "unimol_report.txt")

        smiles_a = "CCO"
        smiles_b = "CC(=O)Oc1ccccc1C(=O)O"
        smiles_bad = "this-is-not-a-molecule"
        atoms_a, atoms_b = 9, 21
        molecules = {smiles_md5(s): s for s in (smiles_a, smiles_b, smiles_bad)}
        atom_counts = {smiles_a: atoms_a, smiles_b: atoms_b}

        def stub_encode(batch):
            """Uni-Mol's real shape: n_atoms rows + one trailing [SEP] row."""
            results = []
            for smiles in batch:
                if smiles == smiles_bad:
                    raise RuntimeError("stubbed Uni-Mol failure (unparseable SMILES)")
                n = atom_counts[smiles]
                results.append((
                    torch.randn(n + 1, UNIMOL_NATIVE_DIM),
                    ["C"] * n + [SEP_TOKEN],
                ))
            return results

        done, already, skipped = run_precompute(
            molecules, stub_encode, cache_dir, batch_size=8, progress_every=1000,
        )

        assert done == 2, f"wrote {done} records, expected 2"
        assert already == 0, f"expected nothing pre-cached, got {already}"
        # The failing molecule must be reported, not fatal.
        assert len(skipped) == 1, f"expected 1 skip, got {len(skipped)}"
        assert skipped[0][0] == smiles_md5(smiles_bad), "wrong molecule skipped"
        assert not os.path.exists(os.path.join(cache_dir, f"{smiles_md5(smiles_bad)}.pt")), \
            "a failed molecule was cached anyway"

        # --- [SEP] dropped: rows == n_atoms, not n_atoms + 1
        for smiles, n_atoms in ((smiles_a, atoms_a), (smiles_b, atoms_b)):
            path = os.path.join(cache_dir, f"{smiles_md5(smiles)}.pt")
            assert os.path.exists(path), f"no cache file named by md5 for {smiles!r}"
            record = torch.load(path, map_location="cpu", weights_only=True)
            assert set(record) == {"features", "n_atoms"}, \
                f"unexpected record keys: {sorted(record)}"
            assert record["features"].dim() == 2, "features must be rank 2"
            assert record["features"].shape[-1] == UNIMOL_NATIVE_DIM, \
                f"width {record['features'].shape[-1]} != {UNIMOL_NATIVE_DIM}"
            assert record["features"].shape[0] == n_atoms, \
                f"[SEP] not dropped: {record['features'].shape[0]} rows != {n_atoms} atoms"
            assert record["n_atoms"] == n_atoms, "n_atoms field wrong"
            assert record["n_atoms"] == record["features"].shape[0], \
                "declared n_atoms != rows"
            assert record["features"].device.type == "cpu", "features must be on cpu"

        # --- a missing / non-[SEP] tail must refuse rather than guess
        try:
            strip_sep_row(torch.randn(5, UNIMOL_NATIVE_DIM), ["C"] * 5)
        except ValueError as exc:
            assert SEP_TOKEN in str(exc), "refusal did not mention [SEP]"
        else:
            raise AssertionError("a tail that is not [SEP] was dropped anyway")

        # --- resumability: a second pass writes nothing new
        done_again, already_again, _ = run_precompute(
            molecules, stub_encode, cache_dir, batch_size=8, progress_every=1000,
        )
        assert done_again == 0, f"re-run wrote {done_again}, expected 0"
        assert already_again == 2, f"re-run saw {already_again} cached, expected 2"

        # --- report is written and names the skip
        text = write_report(report_path, cache_dir, len(molecules), done, already, skipped)
        assert os.path.exists(report_path), "no report written"
        assert smiles_md5(smiles_bad) in text, "report does not name the skipped molecule"

        # --- the real consumer can load what we wrote
        sys.path.insert(0, _REPO_ROOT)
        from drug_3d import Drug3DEncoder

        encoder = Drug3DEncoder(cache_dir=cache_dir, device=torch.device("cpu"))
        features, mask = encoder([smiles_a, smiles_b])
        assert features.shape == (2, 290, UNIMOL_NATIVE_DIM), \
            f"encoder returned {tuple(features.shape)}"
        assert mask.shape == (2, 290), f"mask {tuple(mask.shape)}"
        assert int(mask[0].sum()) == atoms_a, \
            f"mask has {int(mask[0].sum())} atoms, expected {atoms_a}"
        assert int(mask[1].sum()) == atoms_b, \
            f"mask has {int(mask[1].sum())} atoms, expected {atoms_b}"

        print("SELF-TEST PASS")
        return 0

    except Exception as exc:                           # noqa: BLE001
        print(f"SELF-TEST FAILED: {type(exc).__name__}: {exc}")
        return 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --- driver ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute per-atom Uni-Mol embeddings for drug_3d.py."
    )
    parser.add_argument("--self-test", action="store_true",
                        help="stubbed end-to-end check; no gpu, unimol or data")
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--datasets", nargs="+", default=["biosnap", "bindingdb"])
    parser.add_argument("--cache-dir", default=None,
                        help="default: <repo>/cache/unimol")
    parser.add_argument("--device", default=None,
                        help="cuda or cpu; default cuda when available")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N pending molecules (trial runs)")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    cache_dir = args.cache_dir or os.path.join(_REPO_ROOT, "cache", "unimol")
    report_path = os.path.join(os.path.dirname(cache_dir), "unimol_report.txt")
    os.makedirs(cache_dir, exist_ok=True)

    if args.device is not None:
        use_cuda = args.device.startswith("cuda")
    else:
        use_cuda = torch.cuda.is_available()
    print(f"use_cuda: {use_cuda}")

    molecules = load_unique_smiles(args.datasets_root, args.datasets)
    print(f"unique SMILES across {', '.join(args.datasets)}: {len(molecules)}")

    done = already = 0
    skipped = []
    interrupted = False
    try:
        encode = build_unimol_encoder(use_cuda)
        done, already, skipped = run_precompute(
            molecules, encode, cache_dir,
            batch_size=args.batch_size, limit=args.limit,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted - writing report for what completed", flush=True)

    report = write_report(report_path, cache_dir, len(molecules), done, already, skipped)
    print()
    print(report.split("SKIPPED")[0].rstrip())
    print(f"report -> {report_path}")
    if interrupted:
        print("re-run the same command to continue where this stopped")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
