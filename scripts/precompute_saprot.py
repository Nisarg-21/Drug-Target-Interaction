#!/usr/bin/env python
"""Precompute SaProt per-residue embeddings into cache/saprot/.

Reads the structure-aware sequences scripts/run_foldseek.py produced and writes
one record per protein in the format encoders/saprot.py reads:

    cache/saprot/{md5(sequence)}.pt
        {"features": FloatTensor [L, D] (cpu), "length": int L,
         "has_structure": 1 | 0}

The md5 key is of the RAW amino-acid sequence, matching models.py's feature
cache, scripts/fetch_alphafold.py and precompute_protein_3d.py - so a protein
has the same key in cache/esm/, cache/esmif1/ and cache/saprot/ and the three
encoders are directly comparable.

Truncation is max_length=512, identical to ProtBertProteinEncoder
(models.py:167). SaProt tokenises one token per residue, so this is the same
residue budget the ESM-2 baseline gets.

ENVIRONMENT: the training env (transformers). SaProt is an ESM-architecture
masked LM, so unlike ESM-IF1 it needs no separate venv.

    python scripts/precompute_saprot.py --limit 5        # trial
    python scripts/precompute_saprot.py                  # full run
    python scripts/precompute_saprot.py --self-test      # no gpu, no model, no data
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoders.saprot import (            # noqa: E402
    SAPROT_MAX_LENGTH,
    SAPROT_MODEL_ID,
    SaProtModel,
    sequence_cache_key,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_saprot_sequences(path):
    """md5 -> {saprot_sequence, has_structure, ...} from run_foldseek.py."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["records"]


def atomic_save_record(record, path):
    """Temp file + replace, so an interrupted write cannot leave a truncated .pt
    that a resumed run would skip as already done."""
    temp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(record, temp_path)
    os.replace(temp_path, path)


def make_record(features, has_structure):
    features = features.detach().to("cpu").float().contiguous()
    return {
        "features": features,
        "length": int(features.shape[0]),
        "has_structure": int(has_structure),
    }


def run_precompute(records, encode, cache_dir, batch_size=4,
                   progress_every=100, limit=None):
    """Fill cache_dir from run_foldseek.py's records. Returns (done, already, skipped)."""
    os.makedirs(cache_dir, exist_ok=True)

    pending = []
    already = 0
    for key in sorted(records):
        if os.path.exists(os.path.join(cache_dir, f"{key}.pt")):
            already += 1
            continue
        pending.append(key)
    if limit is not None:
        pending = pending[:limit]

    print(f"[saprot] {len(records)} proteins, {already} already cached, "
          f"{len(pending)} to do", flush=True)

    done = 0
    skipped = []
    processed = 0

    for start in range(0, len(pending), batch_size):
        batch_keys = pending[start:start + batch_size]
        batch_seqs = [records[key]["saprot_sequence"] for key in batch_keys]

        try:
            outputs = encode(batch_seqs)
        except Exception as exc:                       # noqa: BLE001
            # Isolate the offender rather than losing the whole batch.
            outputs = []
            for key in batch_keys:
                try:
                    outputs.append(encode([records[key]["saprot_sequence"]])[0])
                except Exception as inner:             # noqa: BLE001
                    outputs.append(RuntimeError(f"{type(inner).__name__}: {inner}"))
            del exc

        for key, features in zip(batch_keys, outputs):
            try:
                if isinstance(features, Exception):
                    raise features
                if features.dim() != 2:
                    raise ValueError(f"expected [L, D], got {tuple(features.shape)}")
                atomic_save_record(
                    make_record(features, records[key].get("has_structure", 0)),
                    os.path.join(cache_dir, f"{key}.pt"),
                )
                done += 1
            except Exception as exc:                   # noqa: BLE001
                skipped.append((key, f"{type(exc).__name__}: {exc}"))

        processed += len(batch_keys)
        if processed % progress_every < batch_size or processed >= len(pending):
            print(f"  [saprot {processed}/{len(pending)}] done={done} "
                  f"skipped={len(skipped)}", flush=True)

    return done, already, skipped


def write_report(report_path, cache_dir, expected_total, done, already, skipped,
                 output_dim=None):
    cached = {
        name[:-3] for name in os.listdir(cache_dir) if name.endswith(".pt")
    } if os.path.isdir(cache_dir) else set()

    lines = []
    lines.append("SaProt precompute report")
    lines.append("=" * 72)
    lines.append(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if output_dim is not None:
        lines.append(f"model output dim: {output_dim}")
    lines.append("")
    lines.append(f"  proteins expected         : {expected_total}")
    lines.append(f"  cached records on disk    : {len(cached)}")
    lines.append(f"  still missing             : {expected_total - len(cached)}")
    lines.append("")
    lines.append(f"  written this run          : {done}")
    lines.append(f"  already cached (skipped)  : {already}")
    lines.append(f"  failed / skipped          : {len(skipped)}")
    lines.append("")
    lines.append(f"SKIPPED ({len(skipped)})")
    lines.append("-" * 72)
    if not skipped:
        lines.append("  (none)")
    for key, reason in skipped:
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
    """Stubbed end-to-end check: no gpu, no SaProt weights, no real data."""
    import shutil
    import tempfile

    root = tempfile.mkdtemp(prefix="precompute_saprot_selftest_")
    try:
        cache_dir = os.path.join(root, "cache", "saprot")
        report_path = os.path.join(root, "cache", "saprot_report.txt")

        seq_a, seq_b = "MKTAY", "MKTAYIA"
        md5_a, md5_b = sequence_cache_key(seq_a), sequence_cache_key(seq_b)
        records = {
            md5_a: {"saprot_sequence": "MdKpTvAnYq", "has_structure": 1,
                    "sequence_length": 5},
            md5_b: {"saprot_sequence": "M#K#T#A#Y#I#A#", "has_structure": 0,
                    "sequence_length": 7},
        }

        def stub_encode(saprot_seqs):
            # One row per residue (SaProt pairs are one token each) + BOS/EOS,
            # mirroring what the real encoder keeps via the attention mask.
            return [torch.randn(len(s) // 2 + 2, 1280) for s in saprot_seqs]

        done, already, skipped = run_precompute(
            records, stub_encode, cache_dir, batch_size=2, progress_every=1000,
        )
        assert done == 2, f"wrote {done}, expected 2"
        assert already == 0 and not skipped, f"already={already} skipped={skipped}"

        # --- records match encoders/saprot.py's expectations
        for md5, seq, flag in ((md5_a, seq_a, 1), (md5_b, seq_b, 0)):
            path = os.path.join(cache_dir, f"{md5}.pt")
            assert os.path.exists(path), f"no cache file named by md5 for {seq!r}"
            record = torch.load(path, map_location="cpu", weights_only=True)
            assert set(record) == {"features", "length", "has_structure"}, \
                f"unexpected keys: {sorted(record)}"
            assert record["features"].dim() == 2, "features must be rank 2"
            assert record["features"].shape[-1] == 1280, "width must be 1280"
            assert record["length"] == record["features"].shape[0], "length != rows"
            assert record["has_structure"] == flag, "has_structure not carried through"
            assert record["features"].device.type == "cpu", "features must be on cpu"

        # --- resumability
        done2, already2, _ = run_precompute(
            records, stub_encode, cache_dir, batch_size=2, progress_every=1000,
        )
        assert done2 == 0 and already2 == 2, f"re-run wrote {done2}, saw {already2}"

        # --- a failing protein is reported, never cached
        bad_md5 = sequence_cache_key("QQQ")
        bad = {bad_md5: {"saprot_sequence": "Q#Q#Q#", "has_structure": 0}}

        def failing_encode(_seqs):
            raise RuntimeError("stubbed SaProt failure")

        done3, _, skipped3 = run_precompute(
            bad, failing_encode, cache_dir, batch_size=2, progress_every=1000,
        )
        assert done3 == 0 and len(skipped3) == 1, "failure was not isolated"
        assert not os.path.exists(os.path.join(cache_dir, f"{bad_md5}.pt")), \
            "a failed protein was cached anyway"

        write_report(report_path, cache_dir, len(records), done, already, skipped)
        assert os.path.exists(report_path), "no report written"

        # --- the real consumer can load what we wrote
        from encoders.saprot import SaProtProteinEncoder

        encoder = SaProtProteinEncoder(cache_dir=cache_dir, device=torch.device("cpu"))
        features, mask = encoder([seq_a, seq_b])
        rows_a = len("MdKpTvAnYq") // 2 + 2
        rows_b = len("M#K#T#A#Y#I#A#") // 2 + 2
        assert features.shape == (2, max(rows_a, rows_b), 1280), \
            f"encoder returned {tuple(features.shape)}"
        assert mask.shape == (2, max(rows_a, rows_b)), f"mask {tuple(mask.shape)}"
        assert int(mask[0].sum()) == rows_a and int(mask[1].sum()) == rows_b, \
            "mask does not match cached lengths"
        assert torch.count_nonzero(features[0, rows_a:]) == 0, "padding not zero-filled"

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
        description="Precompute SaProt per-residue embeddings for encoders/saprot.py."
    )
    parser.add_argument("--self-test", action="store_true",
                        help="stubbed check; no gpu, weights or data")
    parser.add_argument("--saprot-sequences", default=None,
                        help="default: cache/saprot_sequences.json")
    parser.add_argument("--alphafold-root", default="alphafold_data")
    parser.add_argument("--cache-dir", default=None, help="default: <repo>/cache/saprot")
    parser.add_argument("--model-path", default=SAPROT_MODEL_ID,
                        help="HuggingFace id or a local checkpoint directory")
    parser.add_argument("--device", default=None, help="default: cuda if available")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=SAPROT_MAX_LENGTH,
                        help="token budget; 512 matches the ESM-2 baseline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mask-all", action="store_true",
                        help="sequence-only ablation: replace every 3Di slot with '#' "
                             "(use with a separate --cache-dir)")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    cache_dir = args.cache_dir or os.path.join(_REPO_ROOT, "cache", "saprot")
    report_path = os.path.join(os.path.dirname(cache_dir),
                               f"{os.path.basename(os.path.normpath(cache_dir))}_report.txt")
    os.makedirs(cache_dir, exist_ok=True)

    sequences_path = args.saprot_sequences or os.path.join(
        _REPO_ROOT, "cache", "saprot_sequences.json")
    records = load_saprot_sequences(sequences_path)
    if args.mask_all:
        # Even positions are the amino acids, odd positions the 3Di slots.
        records = {
            key: dict(record,
                      saprot_sequence="".join(aa + "#" for aa in record["saprot_sequence"][::2]),
                      has_structure=0)
            for key, record in records.items()
        }
        print("mask-all: every 3Di slot replaced with '#'")
    with_structure = sum(1 for r in records.values() if r.get("has_structure"))
    print(f"proteins: {len(records)}  with 3Di: {with_structure}  "
          f"sequence-only: {len(records) - with_structure}")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    done = already = 0
    skipped = []
    output_dim = None
    interrupted = False
    try:
        model = SaProtModel(model_path=args.model_path, device=device,
                            max_length=args.max_length)
        output_dim = model.output_dim
        print(f"SaProt output dim: {output_dim}"
              + ("" if output_dim == 1280 else "  <-- NOT 1280: a projection will be built"))
        done, already, skipped = run_precompute(
            records, model.encode, cache_dir,
            batch_size=args.batch_size, limit=args.limit,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted - writing report for what completed", flush=True)

    report = write_report(report_path, cache_dir, len(records), done, already,
                          skipped, output_dim)
    print()
    print(report.split("SKIPPED")[0].rstrip())
    print(f"report -> {report_path}")
    if interrupted:
        print("re-run the same command to continue where this stopped")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
