#!/usr/bin/env python
"""Build SaProt structure-aware sequences from the AlphaFold structures.

SaProt consumes a per-residue interleaving of the amino-acid sequence and
Foldseek's 3Di structural alphabet: for residue i the token is

    AA[i].upper() + 3Di[i].lower()

so "MKT" with 3Di "dpv" becomes "MdKpTv". A residue whose structure is unknown
or untrusted gets '#' in the 3Di slot ("Md Kp T#"), which is SaProt's masked
form - the model then reads that position as sequence-only.

Inputs come from scripts/fetch_alphafold.py's outputs on the server:

    alphafold_data/usable_structures.json   md5 -> {accession, structure_file, ...}
    alphafold_data/fallback_list.json       md5 -> {reason, ...}  (no structure)
    alphafold_data/structures/{ACC}.cif     the AlphaFold model
    alphafold_data/plddt/{ACC}.npy          per-residue pLDDT

Every protein in BOTH lists gets an output record: the ones with a clean
structure get real 3Di, the ones without get a fully masked 3Di string so SaProt
still runs on them from sequence alone. That keeps the SaProt cache covering the
same 4,015 proteins the ESM-2 cache does.

Output: a JSON keyed by md5(sequence) - the same key scripts/fetch_alphafold.py,
precompute_protein_3d.py and models.py's feature-cache all use - with a
has_structure flag per protein.

    python scripts/run_foldseek.py --foldseek /path/to/foldseek
    python scripts/run_foldseek.py --foldseek ... --plddt_mask_threshold 70
    python scripts/run_foldseek.py --self-test        # no foldseek, no data
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np


# SaProt's mask token for the structural half of a residue pair.
MASK_3DI = "#"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sequence_md5(sequence):
    """Same keying as models.py:_sequence_cache_key and the rest of the pipeline."""
    return hashlib.md5(sequence.encode("utf-8")).hexdigest()


# --- inputs ----------------------------------------------------------------

def load_sequences_by_md5(datasets_root, dataset_names):
    """md5 -> sequence for every unique protein in the datasets (read-only).

    The selection JSONs store only md5 keys, never the sequence text, so the
    amino-acid half of the SaProt token has to come from the CSVs.
    """
    import pandas as pd

    sequences = {}
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
            frame = pd.read_csv(csv_path, usecols=["Protein"])
            for sequence in frame["Protein"].astype(str):
                sequences.setdefault(sequence_md5(sequence), sequence)

    return sequences


def load_selection(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)["records"]


# --- foldseek --------------------------------------------------------------

def foldseek_3di(structure_path, foldseek_binary, workdir):
    """Run Foldseek over one structure and return its 3Di string.

    Uses `foldseek structureto3didescriptor`, which emits one TSV row per chain:
        name \t amino-acid sequence \t 3Di sequence \t ...
    AlphaFold models are single-chain, so exactly one row is expected.

    Returns (aa_from_structure, three_di).
    """
    output_path = os.path.join(workdir, "3di.tsv")
    command = [
        foldseek_binary, "structureto3didescriptor",
        "--threads", "1",
        "--chain-name-mode", "1",
        structure_path, output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"foldseek exited {result.returncode}: "
            f"{(result.stderr or result.stdout or '').strip()[:400]}"
        )
    if not os.path.exists(output_path):
        raise RuntimeError("foldseek produced no descriptor file")

    rows = []
    with open(output_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) >= 3:
                rows.append((fields[1], fields[2]))

    if not rows:
        raise RuntimeError("foldseek descriptor file had no usable rows")
    if len(rows) > 1:
        # AlphaFold models are single-chain; more than one means we are not
        # looking at what we think we are.
        raise RuntimeError(f"expected 1 chain, foldseek reported {len(rows)}")

    aa_sequence, three_di = rows[0]
    if len(aa_sequence) != len(three_di):
        raise RuntimeError(
            f"foldseek returned {len(aa_sequence)} residues but "
            f"{len(three_di)} 3Di tokens"
        )
    return aa_sequence, three_di


# --- SaProt sequence construction ------------------------------------------

def build_saprot_sequence(sequence, three_di):
    """Interleave AA (upper) and 3Di (lower) into SaProt's token string."""
    if len(three_di) != len(sequence):
        raise ValueError(
            f"3Di length {len(three_di)} != sequence length {len(sequence)}"
        )
    return "".join(
        aa.upper() + (st if st == MASK_3DI else st.lower())
        for aa, st in zip(sequence, three_di)
    )


def masked_saprot_sequence(sequence):
    """Sequence-only SaProt input: every structural slot masked."""
    return "".join(aa.upper() + MASK_3DI for aa in sequence)


def apply_plddt_mask(three_di, plddt, threshold):
    """Replace 3Di with '#' wherever pLDDT is below threshold.

    Off by default. AlphaFold's low-confidence regions are often disordered
    rather than wrong, so masking them is a modelling choice, not a correctness
    fix - which is why raw 3Di is the default and this is opt-in.
    """
    if plddt is None:
        raise ValueError("pLDDT masking requested but no pLDDT array was loaded")
    if len(plddt) != len(three_di):
        raise ValueError(
            f"pLDDT length {len(plddt)} != 3Di length {len(three_di)}"
        )
    return "".join(
        MASK_3DI if score < threshold else token
        for token, score in zip(three_di, plddt)
    )


# --- driver pieces ---------------------------------------------------------

def load_existing(output_path):
    """Previous run's records, for resumability."""
    if not os.path.exists(output_path):
        return {}
    try:
        with open(output_path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("records", {})
    except (ValueError, OSError):
        return {}


def atomic_write_json(payload, path):
    temp_path = f"{path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temp_path, path)


def save_records(records, output_path, threshold):
    atomic_write_json({
        "schema_version": 1,
        "generated_by": "scripts/run_foldseek.py",
        "key": "md5(sequence)",
        "mask_token": MASK_3DI,
        "plddt_mask_threshold": threshold,
        "count": len(records),
        "records": records,
    }, output_path)


def process_all(sequences, usable, fallback, foldseek_binary, alphafold_root,
                output_path, threshold=None, limit=None, progress_every=100,
                foldseek_fn=None):
    """Build SaProt sequences for every protein in both selection lists."""
    records = load_existing(output_path)
    already = len(records)

    pending = []
    for key in sorted(set(usable) | set(fallback)):
        if key in records:
            continue
        pending.append(key)
    if limit is not None:
        pending = pending[:limit]

    print(f"[foldseek] {len(usable)} usable + {len(fallback)} fallback, "
          f"{already} already done, {len(pending)} to do", flush=True)

    run_foldseek = foldseek_fn if foldseek_fn is not None else foldseek_3di
    workdir = tempfile.mkdtemp(prefix="foldseek_")
    demoted = []

    try:
        for position, key in enumerate(pending, start=1):
            sequence = sequences.get(key)
            if sequence is None:
                demoted.append((key, "sequence not found in datasets/"))
                continue

            entry = usable.get(key)
            record = None

            if entry is not None:
                try:
                    structure_path = os.path.join(alphafold_root, entry["structure_file"])
                    _aa, three_di = run_foldseek(structure_path, foldseek_binary, workdir)

                    # The structure must cover exactly our sequence. Anything else
                    # means the 3Di would be pasted onto the wrong residues, which
                    # trains to a plausible but wrong number rather than crashing.
                    if len(three_di) != len(sequence):
                        raise ValueError(
                            f"structure has {len(three_di)} residues, sequence has "
                            f"{len(sequence)}"
                        )

                    if threshold is not None:
                        plddt = np.load(os.path.join(alphafold_root, entry["plddt_file"]))
                        three_di = apply_plddt_mask(three_di, plddt, threshold)

                    record = {
                        "saprot_sequence": build_saprot_sequence(sequence, three_di),
                        "has_structure": 1,
                        "sequence_length": len(sequence),
                        "accession": entry.get("accession"),
                        "masked_residues": three_di.count(MASK_3DI),
                    }
                except Exception as exc:                   # noqa: BLE001
                    # Any structural problem demotes to sequence-only rather than
                    # dropping the protein: SaProt still has something to read.
                    demoted.append((key, f"{type(exc).__name__}: {exc}"))
                    record = None

            if record is None:
                record = {
                    "saprot_sequence": masked_saprot_sequence(sequence),
                    "has_structure": 0,
                    "sequence_length": len(sequence),
                    "accession": (entry or fallback.get(key, {})).get("accession"),
                    "masked_residues": len(sequence),
                }

            records[key] = record

            if position % progress_every == 0 or position == len(pending):
                save_records(records, output_path, threshold)
                with_structure = sum(1 for r in records.values() if r["has_structure"])
                print(f"  [foldseek {position}/{len(pending)}] "
                      f"with_structure={with_structure} demoted={len(demoted)}",
                      flush=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    save_records(records, output_path, threshold)
    return records, already, demoted


def write_report(report_path, records, demoted, threshold):
    with_structure = sum(1 for r in records.values() if r["has_structure"])
    without = len(records) - with_structure

    lines = []
    lines.append("Foldseek / SaProt sequence report")
    lines.append("=" * 72)
    lines.append(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"pLDDT mask threshold: {threshold if threshold is not None else 'off (raw 3Di)'}")
    lines.append("")
    lines.append(f"  proteins total            : {len(records)}")
    lines.append(f"  with real 3Di             : {with_structure}")
    lines.append(f"  sequence-only (all '#')   : {without}")
    lines.append("")
    lines.append(f"DEMOTED TO SEQUENCE-ONLY THIS RUN ({len(demoted)})")
    lines.append("-" * 72)
    if not demoted:
        lines.append("  (none)")
    for key, reason in demoted:
        lines.append(f"  {key}  {reason}")
    lines.append("")

    text = "\n".join(lines)
    temp_path = f"{report_path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    os.replace(temp_path, report_path)
    return text


# --- self-test -------------------------------------------------------------

def self_test():
    """No foldseek, no structures, no datasets."""
    root = tempfile.mkdtemp(prefix="foldseek_selftest_")
    try:
        alphafold_root = os.path.join(root, "alphafold_data")
        os.makedirs(os.path.join(alphafold_root, "structures"), exist_ok=True)
        os.makedirs(os.path.join(alphafold_root, "plddt"), exist_ok=True)
        output_path = os.path.join(root, "saprot_sequences.json")

        seq_ok = "MKTAY"
        seq_bad_len = "MKTAYIA"
        seq_none = "MKTA"
        md5_ok, md5_bad, md5_none = (sequence_md5(s) for s in (seq_ok, seq_bad_len, seq_none))
        sequences = {md5_ok: seq_ok, md5_bad: seq_bad_len, md5_none: seq_none}

        usable = {}
        for md5, seq, acc in ((md5_ok, seq_ok, "P00001"), (md5_bad, seq_bad_len, "P00002")):
            open(os.path.join(alphafold_root, "structures", f"{acc}.cif"), "w").close()
            np.save(os.path.join(alphafold_root, "plddt", f"{acc}.npy"),
                    np.array([95.0, 30.0, 88.0, 20.0, 99.0], dtype=np.float32))
            usable[md5] = {"accession": acc,
                           "structure_file": f"structures/{acc}.cif",
                           "plddt_file": f"plddt/{acc}.npy",
                           "sequence_length": len(seq)}
        fallback = {md5_none: {"accession": None, "reason": "unmapped",
                               "sequence_length": len(seq_none)}}

        def stub_foldseek(structure_path, binary, workdir):
            # P00001 matches the sequence; P00002 deliberately does not.
            if structure_path.endswith("P00001.cif"):
                return "MKTAY", "dpvnq"
            return "MKT", "dpv"

        # --- AA/3Di interleaving
        assert build_saprot_sequence("MKT", "dpv") == "MdKpTv", \
            build_saprot_sequence("MKT", "dpv")
        assert masked_saprot_sequence("MKT") == "M#K#T#", masked_saprot_sequence("MKT")

        records, _, demoted = process_all(
            sequences, usable, fallback, "foldseek-not-needed", alphafold_root,
            output_path, threshold=None, progress_every=1000, foldseek_fn=stub_foldseek,
        )

        assert len(records) == 3, f"expected 3 records, got {len(records)}"

        # --- clean structure -> real 3Di, has_structure 1
        r = records[md5_ok]
        assert r["has_structure"] == 1, "clean structure should have has_structure=1"
        assert r["saprot_sequence"] == "MdKpTvAnYq", r["saprot_sequence"]
        assert r["masked_residues"] == 0, "raw 3Di should mask nothing by default"

        # --- residue-count mismatch -> demoted, fully masked, never misaligned
        r = records[md5_bad]
        assert r["has_structure"] == 0, "length mismatch must demote to sequence-only"
        assert r["saprot_sequence"] == masked_saprot_sequence(seq_bad_len), \
            "demoted protein should be fully masked"
        assert any(k == md5_bad for k, _ in demoted), "mismatch not logged"

        # --- no structure at all -> fully masked, SaProt still runs
        r = records[md5_none]
        assert r["has_structure"] == 0 and r["saprot_sequence"] == "M#K#T#A#", \
            r["saprot_sequence"]

        # --- length invariant: 2 characters per residue, always
        for md5, rec in records.items():
            assert len(rec["saprot_sequence"]) == 2 * rec["sequence_length"], \
                f"{md5}: saprot sequence is not 2 chars per residue"

        # --- pLDDT masking is opt-in and masks exactly the low-confidence slots
        masked = apply_plddt_mask("dpvnq", np.array([95.0, 30.0, 88.0, 20.0, 99.0]), 70)
        assert masked == "d#v#q", masked
        assert build_saprot_sequence("MKTAY", masked) == "MdK#TvA#Yq", \
            build_saprot_sequence("MKTAY", masked)

        # --- resumability: a second pass adds nothing
        records2, already2, _ = process_all(
            sequences, usable, fallback, "foldseek-not-needed", alphafold_root,
            output_path, threshold=None, progress_every=1000, foldseek_fn=stub_foldseek,
        )
        assert already2 == 3, f"re-run saw {already2} existing, expected 3"
        assert len(records2) == 3, "re-run changed the record count"

        # --- output JSON is keyed by md5 and carries has_structure
        with open(output_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["key"] == "md5(sequence)", "wrong key convention"
        assert set(payload["records"]) == {md5_ok, md5_bad, md5_none}, "key mismatch"
        assert all("has_structure" in r for r in payload["records"].values()), \
            "has_structure missing"

        print("SELF-TEST PASS")
        return 0
    except Exception as exc:                               # noqa: BLE001
        print(f"SELF-TEST FAILED: {type(exc).__name__}: {exc}")
        return 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --- main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build SaProt structure-aware sequences via Foldseek 3Di."
    )
    parser.add_argument("--self-test", action="store_true",
                        help="stubbed check; no foldseek, structures or datasets")
    parser.add_argument("--foldseek", default=None,
                        help="path to the foldseek binary (required for a real run)")
    parser.add_argument("--alphafold-root", default="alphafold_data",
                        help="server directory with structures/, plddt/ and the lists")
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--datasets", nargs="+", default=["biosnap", "bindingdb"])
    parser.add_argument("--output", default=None,
                        help="default: <alphafold-root>/saprot_sequences.json")
    parser.add_argument("--plddt_mask_threshold", type=float, default=None,
                        help="OFF by default; below this pLDDT the 3Di token becomes '#'")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.foldseek:
        parser.error("--foldseek is required for a real run (or use --self-test)")

    output_path = args.output or os.path.join(args.alphafold_root, "saprot_sequences.json")
    report_path = os.path.join(os.path.dirname(output_path) or ".", "saprot_sequences_report.txt")

    usable = load_selection(os.path.join(args.alphafold_root, "usable_structures.json"))
    fallback = load_selection(os.path.join(args.alphafold_root, "fallback_list.json"))
    sequences = load_sequences_by_md5(args.datasets_root, args.datasets)
    print(f"sequences: {len(sequences)}  usable: {len(usable)}  fallback: {len(fallback)}")
    print(f"pLDDT masking: "
          f"{'off (raw 3Di)' if args.plddt_mask_threshold is None else args.plddt_mask_threshold}")

    records, already, demoted = process_all(
        sequences, usable, fallback, args.foldseek, args.alphafold_root,
        output_path, threshold=args.plddt_mask_threshold, limit=args.limit,
    )

    report = write_report(report_path, records, demoted, args.plddt_mask_threshold)
    print()
    print(report.split("DEMOTED")[0].rstrip())
    print(f"sequences -> {output_path}")
    print(f"report    -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
