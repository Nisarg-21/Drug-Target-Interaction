#!/usr/bin/env python
"""Maintain result/encoder_ablation_summary.csv for the protein-encoder ablation.

    python scripts/ablation_summary.py append --run-dir result/biosnap_random_saprot_full/seed42 \
        --dataset biosnap --split random --arm saprot_full --seed 42 --runtime-s 21600
    python scripts/ablation_summary.py append ... --status crashed      # no metrics read
    python scripts/ablation_summary.py report                           # print table + flags

Every append re-derives the flags for all rows, because a later arm can turn an
earlier run into an outlier:
    best_epoch<10                 best validation epoch below 10
    auroc_far_below_other_arms    test AUROC more than AUROC_GAP below the best
                                  other arm on the same dataset/split
"""

import argparse
import csv
import os
import sys

import torch

SUMMARY = os.path.join("result", "encoder_ablation_summary.csv")
FIELDS = ["dataset", "split", "arm", "seed", "best_epoch", "auroc", "auprc", "f1",
          "runtime_h", "status", "flags", "note"]
AUROC_GAP = 0.05
MIN_BEST_EPOCH = 10


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    temp_path = f"{path}.tmp.{os.getpid()}"
    with open(temp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def derive_flags(rows):
    for row in rows:
        flags = []
        if row["status"] == "ok":
            if int(row["best_epoch"]) < MIN_BEST_EPOCH:
                flags.append(f"best_epoch<{MIN_BEST_EPOCH}")
            others = [float(o["auroc"]) for o in rows
                      if o is not row and o["status"] == "ok"
                      and (o["dataset"], o["split"]) == (row["dataset"], row["split"])
                      and o["arm"] != row["arm"]]
            if others and float(row["auroc"]) < max(others) - AUROC_GAP:
                flags.append("auroc_far_below_other_arms")
        row["flags"] = ";".join(flags)


def append(args):
    row = {"dataset": args.dataset, "split": args.split, "arm": args.arm, "seed": args.seed,
           "best_epoch": "", "auroc": "", "auprc": "", "f1": "",
           "runtime_h": "" if args.runtime_s is None else f"{args.runtime_s / 3600:.2f}",
           "status": args.status, "flags": "", "note": args.note or ""}
    if args.status == "ok":
        metrics = torch.load(os.path.join(args.run_dir, "result_metrics.pt"),
                             map_location="cpu", weights_only=False)
        test = metrics["test_metrics"]
        row.update({"best_epoch": int(metrics["best_epoch"]),
                    "auroc": f"{float(test['auroc']):.4f}",
                    "auprc": f"{float(test['auprc']):.4f}",
                    "f1": f"{float(test['F1']):.4f}"})
    rows = read_rows(args.summary)
    rows.append(row)
    derive_flags(rows)
    write_rows(args.summary, rows)
    print(f"summary: appended {args.dataset} {args.split} {args.arm} seed{args.seed} "
          f"status={args.status} auroc={row['auroc']} best_epoch={row['best_epoch']} "
          f"flags={rows[-1]['flags'] or '-'}")


def report(args):
    rows = read_rows(args.summary)
    derive_flags(rows)
    if rows:
        write_rows(args.summary, rows)
    for row in rows:
        print("  ".join(f"{row[f]:<{w}}" for f, w in zip(
            ["dataset", "split", "arm", "seed", "best_epoch", "auroc", "auprc", "f1",
             "runtime_h", "status", "flags"], [9, 12, 14, 4, 4, 6, 6, 6, 5, 7, 30])))
    flagged = [r for r in rows if r["flags"]]
    print(f"\n{len(rows)} rows, {len(flagged)} flagged")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", default=SUMMARY, help="summary CSV path")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("append")
    add.add_argument("--run-dir", required=True)
    add.add_argument("--dataset", required=True)
    add.add_argument("--split", required=True)
    add.add_argument("--arm", required=True)
    add.add_argument("--seed", required=True)
    add.add_argument("--runtime-s", type=float, default=None)
    add.add_argument("--status", default="ok", choices=["ok", "crashed", "skipped_exists"])
    add.add_argument("--note", default=None)
    sub.add_parser("report")
    args = parser.parse_args()
    return append(args) if args.command == "append" else report(args)


if __name__ == "__main__":
    sys.exit(main())
