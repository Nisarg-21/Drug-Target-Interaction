"""Read-only inspection of the existing random-split CSVs for a DTI dataset.

Concatenates train/val/test and reports interaction counts and the
per-drug / per-protein degree distribution, which tells us how many
entities are singletons (hard to place in a constrained split).
"""
import argparse
import os

import numpy as np
import pandas as pd


def load(data_dir):
    frames = []
    for split in ("train", "val", "test"):
        path = os.path.join(data_dir, split + ".csv")
        df = pd.read_csv(path)
        print("  {:<10s} {:>8d} rows   {}".format(split, len(df), path))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def describe(counts, entity):
    print("  min/median/max interactions per {}: {} / {} / {}".format(
        entity, int(counts.min()), float(np.median(counts)), int(counts.max())))
    print("  mean interactions per {}: {:.2f}".format(entity, counts.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, choices=["bindingdb", "biosnap"])
    args = parser.parse_args()

    data_dir = os.path.join("datasets", args.data, "random")

    print("=" * 70)
    print("Dataset: {}   ({})".format(args.data, data_dir))
    print("=" * 70)

    print("\n[files]")
    df = load(data_dir)

    print("\n[columns]")
    print("  column names: {}".format(list(df.columns)))
    print("  dtypes:")
    for col, dt in df.dtypes.items():
        print("    {:<12s} {}".format(str(col), dt))
    print("  first row:")
    for col in df.columns:
        val = str(df.iloc[0][col])
        if len(val) > 60:
            val = val[:60] + "... (len {})".format(len(str(df.iloc[0][col])))
        print("    {:<12s} {}".format(str(col), val))

    drug_col, prot_col, label_col = df.columns[0], df.columns[1], df.columns[2]
    print("\n  interpreting -> drug: '{}', protein: '{}', label: '{}'".format(
        drug_col, prot_col, label_col))

    print("\n[totals]")
    print("  total interactions : {}".format(len(df)))
    print("  unique drugs       : {}".format(df[drug_col].nunique()))
    print("  unique proteins    : {}".format(df[prot_col].nunique()))
    print("  duplicate (drug, protein) pairs: {}".format(
        len(df) - len(df.drop_duplicates(subset=[drug_col, prot_col]))))

    print("\n[label distribution: '{}']".format(label_col))
    for value, n in df[label_col].value_counts().sort_index().items():
        print("  {:<6} {:>8d}  ({:.2%})".format(str(value), n, n / len(df)))

    drug_counts = df.groupby(drug_col).size()
    prot_counts = df.groupby(prot_col).size()

    print("\n[interactions per drug]")
    describe(drug_counts, "drug")

    print("\n[interactions per protein]")
    describe(prot_counts, "protein")

    # Distinct partners (dedup pairs) - what actually matters for splitting.
    pairs = df.drop_duplicates(subset=[drug_col, prot_col])
    drug_partners = pairs.groupby(drug_col)[prot_col].nunique()
    prot_partners = pairs.groupby(prot_col)[drug_col].nunique()

    n_drug_single = int((drug_partners == 1).sum())
    n_prot_single = int((prot_partners == 1).sum())

    print("\n[singletons - hard to place in constrained splits]")
    print("  drugs interacting with exactly 1 protein : {} / {}  ({:.2%})".format(
        n_drug_single, len(drug_partners), n_drug_single / len(drug_partners)))
    print("  proteins interacting with exactly 1 drug : {} / {}  ({:.2%})".format(
        n_prot_single, len(prot_partners), n_prot_single / len(prot_partners)))

    print("\n[degree percentiles (distinct partners)]")
    qs = [1, 5, 25, 50, 75, 95, 99]
    print("  drug    : " + "  ".join(
        "p{}={:.0f}".format(q, np.percentile(drug_partners, q)) for q in qs))
    print("  protein : " + "  ".join(
        "p{}={:.0f}".format(q, np.percentile(prot_partners, q)) for q in qs))
    print()


if __name__ == "__main__":
    main()
