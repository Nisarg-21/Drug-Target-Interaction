#!/usr/bin/env python
"""Re-score a finished run's test set on the proteins that have real 3Di only.

The SaProt full-3Di arm differs from the sequence-only arm only on proteins with a
structure (has_structure=1 in cache/saprot_sequences.json; 3,648 of 4,015). The 367
sequence-only proteins get the same '#' input in both arms and dilute the
comparison, so this restricts the test pairs to structured proteins and recomputes
AUROC, AUPRC and F1.

Nothing in the training code changes: the model is rebuilt from the run's own saved
config (result_metrics.pt), loaded from best_model.pth, and F1 uses the run's own
validation-selected threshold (test_metrics['thred_optim']), as trainer.test() does.
The full test set is scored too and compared to the stored metrics, which checks the
rebuild reproduces the original evaluation.

    python scripts/eval_structure_subset.py --run-dir result/biosnap_random_saprot_full/seed42 \
        --dataset biosnap --split random --arm saprot_full --seed 42
"""

import argparse
import csv
import hashlib
import json
import os
import sys

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import DTIDataset          # noqa: E402
from models import CMA, binary_cross_entropy  # noqa: E402
from utils import graph_collate_func       # noqa: E402

OUTPUT = os.path.join("result", "encoder_ablation_structure_subset.csv")
FIELDS = ["dataset", "split", "arm", "seed", "pairs_full", "pairs_subset", "proteins_subset",
          "auroc_subset", "auprc_subset", "f1_subset", "auroc_full_recomputed",
          "auroc_full_stored", "threshold"]


def score(model, frame, batch_size, device):
    if frame.empty:
        return [], []
    frame = frame.reset_index(drop=True)
    loader = DataLoader(DTIDataset(frame.index.values, frame), batch_size=batch_size,
                        shuffle=False, drop_last=False, collate_fn=graph_collate_func)
    labels, probs = [], []
    model.eval()
    with torch.no_grad():
        for v_d, smiles, proteins, y in loader:
            score_out, _ = model(v_d.to(device), smiles, proteins, mode="eval")
            n, _ = binary_cross_entropy(score_out, y.float().to(device))
            labels.extend(y.tolist())
            probs.extend(n.cpu().tolist())
    return labels, probs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--saprot-sequences", default=os.path.join("cache", "saprot_sequences.json"))
    parser.add_argument("--esm_path", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--chemberta_path", default="./chemberta_model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=OUTPUT, help="CSV the row is appended to")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    metrics = torch.load(os.path.join(args.run_dir, "result_metrics.pt"),
                         map_location="cpu", weights_only=False)
    cfg = metrics["config"]
    threshold = float(metrics["test_metrics"]["thred_optim"])

    with open(args.saprot_sequences, encoding="utf-8") as handle:
        structured = {key for key, record in json.load(handle)["records"].items()
                      if record["has_structure"]}

    test = pd.read_csv(os.path.join("datasets", args.dataset, args.split, "test.csv"))
    keys = test["Protein"].astype(str).map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
    subset = test[keys.isin(structured)]

    model = CMA(protbert_model_path=args.esm_path, chemberta_model_path=args.chemberta_path,
                device=device, **cfg).to(device)
    state = torch.load(os.path.join(args.run_dir, "best_model.pth"), map_location=device,
                       weights_only=True)
    model.load_state_dict(state)

    batch_size = int(cfg["SOLVER"]["BATCH_SIZE"])
    full_y, full_p = score(model, test, batch_size, device)
    sub_y, sub_p = score(model, subset, batch_size, device)

    row = {
        "dataset": args.dataset, "split": args.split, "arm": args.arm, "seed": args.seed,
        "pairs_full": len(test), "pairs_subset": len(subset),
        "proteins_subset": int(keys[keys.isin(structured)].nunique()),
        "auroc_subset": f"{roc_auc_score(sub_y, sub_p):.4f}",
        "auprc_subset": f"{average_precision_score(sub_y, sub_p):.4f}",
        "f1_subset": f"{f1_score(sub_y, [int(p >= threshold) for p in sub_p], zero_division=0):.4f}",
        "auroc_full_recomputed": f"{roc_auc_score(full_y, full_p):.4f}",
        "auroc_full_stored": f"{float(metrics['test_metrics']['auroc']):.4f}",
        "threshold": f"{threshold:.4f}",
    }
    if abs(float(row["auroc_full_recomputed"]) - float(row["auroc_full_stored"])) > 1e-3:
        print("WARNING: recomputed full-test AUROC differs from the stored value; "
              "the rebuild does not reproduce the original evaluation")

    new_file = not os.path.exists(args.output)
    with open(args.output, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
