#!/usr/bin/env python
"""Precompute per-residue protein embeddings into the cache protein_3d.py reads.

Writes one record per sequence to cache/esmif1/{md5(sequence)}.pt in exactly the
format Protein3DEncoder._load_record validates (protein_3d.py:76-113):

    {"features": FloatTensor [L, D_native] (cpu), "length": int L,
     "has_structure": 1 | 0}

Two sources, selected by the lists scripts/fetch_alphafold.py produced:

    alphafold_data/usable_structures.json  (3610)  ESM-IF1 on the AlphaFold
                                                   model -> 512-wide,
                                                   has_structure=1
    alphafold_data/fallback_list.json      ( 405)  ESM-2, baseline settings
                                                   -> 1280-wide,
                                                   has_structure=0

Per-residue pLDDT for the ESM-IF1 proteins is copied to
cache/esmif1_plddt/{md5}.npy. Nothing masks by pLDDT today; it is cached now so
that masking can be added later without re-running the whole precompute.

TWO STAGES, TWO ENVIRONMENTS
ESM-IF1 and the training stack pin incompatible numpy versions, so the stages
must not share an interpreter:

    --stage esmif1   run under  ~/test_3d_env   (fair-esm + biotite + torch)
    --stage esm2     run under  cmadti_env      (the training env: transformers)

Each stage is independently resumable, so they can run in either order, on
different days, and be interrupted freely.

    python precompute_protein_3d.py --stage esmif1
    python precompute_protein_3d.py --stage esm2 --esm2-path /path/to/esm2_t33_650M_UR50D
    python precompute_protein_3d.py --self-test      # no gpu, no models, no data

NOTE ON SEQUENCES
Neither selection JSON stores the sequence string - only md5(sequence) keys. The
ESM-2 stage needs the real text to tokenize, so both stages rebuild the
md5 -> sequence map from the dataset CSVs with the same keying the fetch script
and protein_3d.py use.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch


# Widths are fixed by protein_3d.py's validation; they are not tunable here.
ESMIF1_NATIVE_DIM = 512
ESM2_NATIVE_DIM = 1280

# Baseline ESM-2 tokenisation, identical to ProtBertProteinEncoder (models.py:167).
ESM2_MAX_LENGTH = 512

# AlphaFold models are single-chain.
ALPHAFOLD_CHAIN_ID = "A"

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def sequence_md5(sequence):
    """Same keying as protein_3d.py:36 and scripts/fetch_alphafold.py."""
    return hashlib.md5(sequence.encode("utf-8")).hexdigest()


# --- inputs ----------------------------------------------------------------

def load_sequences_by_md5(datasets_root, dataset_names):
    """md5 -> sequence for every unique protein in the datasets (read-only)."""
    import pandas as pd

    sequences = {}
    for name in dataset_names:
        dataset_dir = os.path.join(datasets_root, name)
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
    """Read a fetch_alphafold.py selection list and return its records dict."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)["records"]


# --- atomic writes ---------------------------------------------------------

def atomic_save_record(record, path):
    """Temp file + replace, so an interrupted write cannot leave a truncated .pt
    that a resumed run would skip as already done."""
    temp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(record, temp_path)
    os.replace(temp_path, path)


def atomic_save_npy(array, path):
    # np.save appends .npy unless the name already ends in it, so the temp name
    # carries the suffix and the replace target stays unambiguous.
    temp_path = f"{path}.tmp.{os.getpid()}.npy"
    np.save(temp_path, array)
    os.replace(temp_path, path)


def make_record(features, has_structure):
    """Build a record in exactly protein_3d.py's expected shape."""
    features = features.detach().to("cpu").float().contiguous()
    return {
        "features": features,
        "length": int(features.shape[0]),
        "has_structure": int(has_structure),
    }


# --- encoders (lazy imports: absent in the self-test env) -------------------

def build_esmif1_encoder(device):
    """Return encode(cif_path) -> FloatTensor [L, 512] on cpu.

    Imported here rather than at module scope so --self-test runs in an
    environment with neither fair-esm nor biotite installed.
    """
    import esm
    import esm.inverse_folding as inverse_folding

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.to(device).eval()
    batch_converter = inverse_folding.util.CoordBatchConverter(alphabet)

    def encode(cif_path):
        structure = inverse_folding.util.load_structure(cif_path, ALPHAFOLD_CHAIN_ID)
        coords, _native_seq = inverse_folding.util.extract_coords_from_structure(structure)

        # get_encoder_output() builds its batch on CPU and never moves it, so on a
        # cuda model the encoder would receive cpu tensors. Converting with an
        # explicit device= puts coords/confidence/padding_mask on the gpu first.
        batch = [(coords, None, None)]
        coords_t, confidence, _strs, _tokens, padding_mask = batch_converter(
            batch, device=device
        )

        with torch.no_grad():
            encoder_out = model.encoder.forward(
                coords_t, padding_mask, confidence, return_all_hiddens=False
            )

        # encoder_out['encoder_out'][0] is [L+2, B, 512]: one leading and one
        # trailing boundary token. Dropping both leaves one row per residue, so a
        # 142-residue chain yields (142, 512).
        representations = encoder_out["encoder_out"][0][1:-1, 0]
        return representations.detach().to("cpu")

    return encode


def build_esm2_encoder(model_path, device):
    """Return encode(list[str]) -> list of FloatTensor [L, 1280] on cpu.

    Mirrors ProtBertProteinEncoder's uncached path exactly (models.py:135,
    :167-178): AutoModelForMaskedLM, padding=True, truncation=True,
    max_length=512, last hidden state, real rows taken via the attention mask.
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(device).eval()

    def encode(sequences):
        encoded = tokenizer(
            sequences, padding=True, truncation=True,
            return_tensors="pt", max_length=ESM2_MAX_LENGTH,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)

        features = outputs.hidden_states[-1]
        attention_mask = encoded["attention_mask"]

        # Keep the mask==1 rows. Those include ESM-2's BOS/EOS, exactly as the
        # baseline's own cache stores them (models.py:86-127), so a fallback
        # record is byte-comparable with what the ESM-2 path would have produced.
        results = []
        for row in range(features.shape[0]):
            keep = attention_mask[row].bool()
            results.append(features[row][keep].detach().to("cpu"))
        return results

    return encode


# --- stages ----------------------------------------------------------------

def run_esmif1_stage(selection, sequences, encode, cache_dir, plddt_cache_dir,
                     alphafold_root, progress_every=50, limit=None):
    """ESM-IF1 over the usable structures. Returns (done, skipped, skip_details)."""
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(plddt_cache_dir, exist_ok=True)

    pending = []
    already = 0
    for key in sorted(selection):
        if os.path.exists(os.path.join(cache_dir, f"{key}.pt")):
            already += 1
            continue
        pending.append(key)
    if limit is not None:
        pending = pending[:limit]

    print(f"[esmif1] {len(selection)} usable, {already} already cached, "
          f"{len(pending)} to do", flush=True)

    done = 0
    skipped = []
    for position, key in enumerate(pending, start=1):
        entry = selection[key]
        accession = entry["accession"]
        cif_path = os.path.join(alphafold_root, entry["structure_file"])
        expected_length = int(entry["sequence_length"])

        try:
            features = encode(cif_path)

            if features.dim() != 2 or features.shape[-1] != ESMIF1_NATIVE_DIM:
                raise ValueError(
                    f"expected [L, {ESMIF1_NATIVE_DIM}], got {tuple(features.shape)}"
                )
            # A misaligned record is worse than a missing one: protein_3d.py would
            # load it happily and every residue index would be off. Skip instead.
            if features.shape[0] != expected_length:
                raise ValueError(
                    f"length {features.shape[0]} != sequence length {expected_length}"
                )

            plddt_source = os.path.join(alphafold_root, entry["plddt_file"])
            plddt = np.load(plddt_source)
            if plddt.shape[0] != features.shape[0]:
                raise ValueError(
                    f"pLDDT length {plddt.shape[0]} != residues {features.shape[0]}"
                )

            atomic_save_record(make_record(features, has_structure=1),
                               os.path.join(cache_dir, f"{key}.pt"))
            atomic_save_npy(plddt.astype(np.float32),
                            os.path.join(plddt_cache_dir, f"{key}.npy"))
            done += 1

        except Exception as exc:                       # noqa: BLE001 - reported, not raised
            skipped.append((key, accession, f"{type(exc).__name__}: {exc}"))

        if position % progress_every == 0 or position == len(pending):
            print(f"  [esmif1 {position}/{len(pending)}] done={done} "
                  f"skipped={len(skipped)}", flush=True)

    return done, already, skipped


def run_esm2_stage(selection, sequences, encode, cache_dir, batch_size=8,
                   progress_every=50, limit=None):
    """ESM-2 over the fallback list. Returns (done, already, skip_details)."""
    os.makedirs(cache_dir, exist_ok=True)

    pending = []
    already = 0
    missing_sequence = []
    for key in sorted(selection):
        if os.path.exists(os.path.join(cache_dir, f"{key}.pt")):
            already += 1
            continue
        if key not in sequences:
            missing_sequence.append((key, None, "sequence not found in datasets/"))
            continue
        pending.append(key)
    if limit is not None:
        pending = pending[:limit]

    print(f"[esm2] {len(selection)} fallback, {already} already cached, "
          f"{len(pending)} to do", flush=True)

    done = 0
    skipped = list(missing_sequence)
    processed = 0

    for start in range(0, len(pending), batch_size):
        batch_keys = pending[start:start + batch_size]
        batch_seqs = [sequences[key] for key in batch_keys]

        try:
            outputs = encode(batch_seqs)
        except Exception as exc:                       # noqa: BLE001
            for key in batch_keys:
                skipped.append((key, None, f"{type(exc).__name__}: {exc}"))
            processed += len(batch_keys)
            continue

        for key, features in zip(batch_keys, outputs):
            try:
                if features.dim() != 2 or features.shape[-1] != ESM2_NATIVE_DIM:
                    raise ValueError(
                        f"expected [L, {ESM2_NATIVE_DIM}], got {tuple(features.shape)}"
                    )
                atomic_save_record(make_record(features, has_structure=0),
                                   os.path.join(cache_dir, f"{key}.pt"))
                done += 1
            except Exception as exc:                   # noqa: BLE001
                skipped.append((key, None, f"{type(exc).__name__}: {exc}"))

        processed += len(batch_keys)
        if processed % progress_every < batch_size or processed >= len(pending):
            print(f"  [esm2 {processed}/{len(pending)}] done={done} "
                  f"skipped={len(skipped)}", flush=True)

    return done, already, skipped


# --- report ----------------------------------------------------------------

def write_report(report_path, cache_dir, usable_count, fallback_count, stage_results):
    """Rewrite the report from what is actually on disk plus this run's skips."""
    cached = {
        name[:-3] for name in os.listdir(cache_dir) if name.endswith(".pt")
    } if os.path.isdir(cache_dir) else set()

    expected_total = usable_count + fallback_count

    lines = []
    lines.append("Protein 3D precompute report")
    lines.append("=" * 72)
    lines.append(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("EXPECTED")
    lines.append(f"  ESM-IF1 (has_structure=1) : {usable_count}")
    lines.append(f"  ESM-2   (has_structure=0) : {fallback_count}")
    lines.append(f"  total                     : {expected_total}")
    lines.append("")
    lines.append("ON DISK")
    lines.append(f"  cached records            : {len(cached)}")
    lines.append(f"  still missing             : {expected_total - len(cached)}")
    lines.append("")

    for stage, result in stage_results.items():
        done, already, skipped = result
        lines.append(f"STAGE {stage}")
        lines.append(f"  written this run          : {done}")
        lines.append(f"  already cached (skipped)  : {already}")
        lines.append(f"  failed / skipped          : {len(skipped)}")
        if skipped:
            lines.append("  " + "-" * 68)
            for key, accession, reason in skipped:
                label = f"{key}  {accession}" if accession else key
                lines.append(f"    {label}  {reason}")
        lines.append("")

    text = "\n".join(lines)
    temp_path = f"{report_path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    os.replace(temp_path, report_path)
    return text


# --- self-test -------------------------------------------------------------

def self_test():
    """Exercise both stages with stubbed encoders: no gpu, no models, no data."""
    import shutil
    import tempfile

    root = tempfile.mkdtemp(prefix="precompute_selftest_")
    try:
        cache_dir = os.path.join(root, "cache", "esmif1")
        plddt_cache_dir = os.path.join(root, "cache", "esmif1_plddt")
        alphafold_root = os.path.join(root, "alphafold_data")
        os.makedirs(os.path.join(alphafold_root, "plddt"), exist_ok=True)
        os.makedirs(os.path.join(alphafold_root, "structures"), exist_ok=True)

        # Three fake proteins: two with "structures", one fallback.
        seq_a = "MKTAYIAKQR"          # 10 residues
        seq_b = "MKTAYIAKQRQISFVKSH"  # 18 residues
        seq_c = "MKTAY"               # 5  residues, fallback
        md5_a, md5_b, md5_c = (sequence_md5(s) for s in (seq_a, seq_b, seq_c))
        sequences = {md5_a: seq_a, md5_b: seq_b, md5_c: seq_c}

        usable = {}
        for md5, seq, accession in ((md5_a, seq_a, "P00001"), (md5_b, seq_b, "P00002")):
            np.save(os.path.join(alphafold_root, "plddt", f"{accession}.npy"),
                    np.random.rand(len(seq)).astype(np.float32) * 100)
            open(os.path.join(alphafold_root, "structures", f"{accession}.cif"), "w").close()
            usable[md5] = {
                "accession": accession,
                "structure_file": f"structures/{accession}.cif",
                "plddt_file": f"plddt/{accession}.npy",
                "sequence_length": len(seq),
                "plddt_length": len(seq),
            }
        fallback = {md5_c: {"accession": None, "reason": "unmapped",
                            "sequence_length": len(seq_c)}}

        # Stubs standing in for the real encoders, returning correctly shaped noise.
        lengths_by_cif = {
            os.path.join(alphafold_root, f"structures/{a}.cif"): n
            for a, n in (("P00001", len(seq_a)), ("P00002", len(seq_b)))
        }

        def stub_esmif1(cif_path):
            return torch.randn(lengths_by_cif[cif_path], ESMIF1_NATIVE_DIM)

        def stub_esm2(seqs):
            # +2 mirrors ESM-2's BOS/EOS, which the real path keeps via the mask.
            return [torch.randn(len(s) + 2, ESM2_NATIVE_DIM) for s in seqs]

        done_if1, _, skipped_if1 = run_esmif1_stage(
            usable, sequences, stub_esmif1, cache_dir, plddt_cache_dir,
            alphafold_root, progress_every=1000,
        )
        done_esm2, _, skipped_esm2 = run_esm2_stage(
            fallback, sequences, stub_esm2, cache_dir,
            batch_size=2, progress_every=1000,
        )

        assert done_if1 == 2, f"esmif1 wrote {done_if1}, expected 2"
        assert not skipped_if1, f"esmif1 skipped: {skipped_if1}"
        assert done_esm2 == 1, f"esm2 wrote {done_esm2}, expected 1"
        assert not skipped_esm2, f"esm2 skipped: {skipped_esm2}"

        # --- records satisfy protein_3d.py's contract
        for md5, seq, width, flag in (
            (md5_a, seq_a, ESMIF1_NATIVE_DIM, 1),
            (md5_b, seq_b, ESMIF1_NATIVE_DIM, 1),
            (md5_c, seq_c, ESM2_NATIVE_DIM, 0),
        ):
            path = os.path.join(cache_dir, f"{md5}.pt")
            assert os.path.exists(path), f"no cache file named by md5 for {seq!r}"
            record = torch.load(path, map_location="cpu", weights_only=True)
            assert set(record) == {"features", "length", "has_structure"}, \
                f"unexpected record keys: {sorted(record)}"
            assert record["features"].dim() == 2, "features must be rank 2"
            assert record["features"].shape[-1] == width, \
                f"width {record['features'].shape[-1]} != {width}"
            assert record["has_structure"] == flag, "has_structure flag wrong"
            assert record["length"] == record["features"].shape[0], \
                "declared length != rows"
            assert record["features"].device.type == "cpu", "features must be on cpu"

        # --- ESM-IF1 rows are one per residue, and pLDDT was cached alongside
        record_a = torch.load(os.path.join(cache_dir, f"{md5_a}.pt"),
                              map_location="cpu", weights_only=True)
        assert record_a["length"] == len(seq_a), \
            f"esmif1 length {record_a['length']} != residues {len(seq_a)}"
        plddt_a = np.load(os.path.join(plddt_cache_dir, f"{md5_a}.npy"))
        assert plddt_a.shape[0] == len(seq_a), "cached pLDDT length wrong"
        assert not os.path.exists(os.path.join(plddt_cache_dir, f"{md5_c}.npy")), \
            "fallback protein should have no pLDDT"

        # --- a length mismatch must be skipped, never cached
        bad_md5 = sequence_md5("MKTAYIAKQRQISFVKSHFSR")
        bad = {bad_md5: {"accession": "P00003",
                         "structure_file": "structures/P00001.cif",
                         "plddt_file": "plddt/P00001.npy",
                         "sequence_length": 999, "plddt_length": 999}}
        done_bad, _, skipped_bad = run_esmif1_stage(
            bad, sequences, stub_esmif1, cache_dir, plddt_cache_dir,
            alphafold_root, progress_every=1000,
        )
        assert done_bad == 0 and len(skipped_bad) == 1, "mismatch was not skipped"
        assert not os.path.exists(os.path.join(cache_dir, f"{bad_md5}.pt")), \
            "a misaligned record was cached"

        # --- resumability: a second pass writes nothing new
        again, already, _ = run_esmif1_stage(
            usable, sequences, stub_esmif1, cache_dir, plddt_cache_dir,
            alphafold_root, progress_every=1000,
        )
        assert again == 0 and already == 2, "re-run did not skip cached records"

        # --- the real consumer can load what we wrote
        sys.path.insert(0, _REPO_ROOT)
        from protein_3d import Protein3DEncoder

        encoder = Protein3DEncoder(cache_dir=cache_dir, device=torch.device("cpu"))
        features, mask = encoder([seq_a, seq_b, seq_c])
        expected_rows = max(len(seq_a), len(seq_b), len(seq_c) + 2)
        assert features.shape == (3, expected_rows, 1280), \
            f"encoder returned {tuple(features.shape)}"
        assert mask.shape == (3, expected_rows), f"mask {tuple(mask.shape)}"
        assert int(mask[0].sum()) == len(seq_a), "mask does not match residue count"
        assert features.requires_grad, "projection is not live"

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
        description="Precompute per-residue protein embeddings for protein_3d.py."
    )
    parser.add_argument("--stage", choices=["esmif1", "esm2"],
                        help="esmif1 runs under ~/test_3d_env; esm2 under cmadti_env")
    parser.add_argument("--self-test", action="store_true",
                        help="stubbed end-to-end check; no gpu, models or data")
    parser.add_argument("--alphafold-root", default="alphafold_data",
                        help="directory holding structures/, plddt/ and the lists")
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--datasets", nargs="+", default=["biosnap", "bindingdb"])
    parser.add_argument("--cache-dir", default=None,
                        help="default: <repo>/cache/esmif1")
    parser.add_argument("--plddt-cache-dir", default=None,
                        help="default: <repo>/cache/esmif1_plddt")
    parser.add_argument("--esm2-path", default=None,
                        help="local esm2_t33_650M_UR50D directory (required for --stage esm2)")
    parser.add_argument("--device", default=None, help="default: cuda if available")
    parser.add_argument("--batch-size", type=int, default=8, help="ESM-2 batch size")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N pending sequences (trial runs)")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.stage:
        parser.error("one of --stage {esmif1,esm2} or --self-test is required")

    cache_dir = args.cache_dir or os.path.join(_REPO_ROOT, "cache", "esmif1")
    plddt_cache_dir = args.plddt_cache_dir or os.path.join(_REPO_ROOT, "cache", "esmif1_plddt")
    report_path = os.path.join(os.path.dirname(cache_dir), "esmif1_report.txt")
    os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")

    usable = load_selection(os.path.join(args.alphafold_root, "usable_structures.json"))
    fallback = load_selection(os.path.join(args.alphafold_root, "fallback_list.json"))
    sequences = load_sequences_by_md5(args.datasets_root, args.datasets)
    print(f"sequences from datasets: {len(sequences)}  "
          f"usable: {len(usable)}  fallback: {len(fallback)}")

    stage_results = {}
    interrupted = False
    try:
        if args.stage == "esmif1":
            encode = build_esmif1_encoder(device)
            stage_results["esmif1"] = run_esmif1_stage(
                usable, sequences, encode, cache_dir, plddt_cache_dir,
                args.alphafold_root, limit=args.limit,
            )
        else:
            if not args.esm2_path:
                parser.error("--stage esm2 requires --esm2-path")
            encode = build_esm2_encoder(args.esm2_path, device)
            stage_results["esm2"] = run_esm2_stage(
                fallback, sequences, encode, cache_dir,
                batch_size=args.batch_size, limit=args.limit,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted - writing report for what completed", flush=True)

    report = write_report(report_path, cache_dir, len(usable), len(fallback), stage_results)
    print()
    print(report.split("STAGE")[0].rstrip())
    print(f"report -> {report_path}")
    if interrupted:
        print("re-run the same command to continue where this stopped")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
