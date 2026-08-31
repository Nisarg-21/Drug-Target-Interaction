"""Generate cold-drug and cold-protein splits from the existing random splits.

Reads datasets/{data}/random/{train,val,test}.csv, concatenates them,
deduplicates on (SMILES, Protein), then holds out entities (not rows):
unique drugs -- or unique proteins -- are partitioned 70/10/20 and every
interaction follows its entity into that fold. The row-level percentages
therefore drift from 70/10/20; that is expected and is reported below.

Read-only with respect to the existing random/ CSVs. Writes:
    datasets/{data}/cold_drug/{train,val,test}.csv
    datasets/{data}/cold_protein/{train,val,test}.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

DRUG_COL = "SMILES"
PROT_COL = "Protein"
LABEL_COL = "Y"
COLUMNS = [DRUG_COL, PROT_COL, LABEL_COL]

FOLDS = ("train", "val", "test")
FRACS = (0.7, 0.1, 0.2)


def load_random(data):
    data_dir = os.path.join("datasets", data, "random")
    frames = []
    print("[load]")
    for split in FOLDS:
        path = os.path.join(data_dir, split + ".csv")
        df = pd.read_csv(path)
        print("  {:<6s} {:>7d} rows   {}".format(split, len(df), path))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print("  {:<6s} {:>7d} rows".format("total", len(df)))
    return df[COLUMNS]


def deduplicate(df):
    """Drop repeated (drug, protein) rows; drop ambiguous pairs entirely."""
    print("\n[dedup]")
    n_before = len(df)
    print("  rows before dedup            : {}".format(n_before))

    key = [DRUG_COL, PROT_COL]
    n_labels = df.groupby(key)[LABEL_COL].nunique()
    ambiguous = n_labels[n_labels > 1].index

    if len(ambiguous) > 0:
        mask = pd.MultiIndex.from_frame(df[key]).isin(ambiguous)
        n_ambiguous_rows = int(mask.sum())
        df = df[~mask]
    else:
        n_ambiguous_rows = 0

    print("  contradictory pairs dropped  : {} pairs ({} rows) -- labels disagreed".format(
        len(ambiguous), n_ambiguous_rows))

    n_pre_exact = len(df)
    df = df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    print("  exact duplicate rows dropped : {}".format(n_pre_exact - len(df)))
    print("  rows after dedup             : {}  (removed {} total)".format(
        len(df), n_before - len(df)))
    print("  unique drugs   : {}".format(df[DRUG_COL].nunique()))
    print("  unique proteins: {}".format(df[PROT_COL].nunique()))
    return df


def partition_entities(entities, rng):
    """Shuffle entities and cut them 70/10/20 by count."""
    entities = np.array(sorted(entities), dtype=object)
    rng.shuffle(entities)
    n = len(entities)
    n_train = int(round(FRACS[0] * n))
    n_val = int(round(FRACS[1] * n))
    return {
        "train": set(entities[:n_train]),
        "val": set(entities[n_train:n_train + n_val]),
        "test": set(entities[n_train + n_val:]),
    }


def cold_split(df, cold_col, seed):
    """Split rows by holding out the entities in `cold_col`."""
    rng = np.random.RandomState(seed)
    buckets = partition_entities(df[cold_col].unique(), rng)
    return {f: df[df[cold_col].isin(buckets[f])].reset_index(drop=True) for f in FOLDS}


def check_disjoint(folds, col, label):
    """Hard assertions -- the whole point of a cold split."""
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(folds[a][col]) & set(folds[b][col])
        assert not overlap, "{}: {} leak between {} and {} ({} shared)".format(
            label, col, a, b, len(overlap))
    print("  assertions passed: {} disjoint across train/val, train/test, val/test".format(col))


def report(folds, label):
    total = sum(len(folds[f]) for f in FOLDS)
    print("\n  {}".format(label))
    print("  {:<6s} {:>8s} {:>8s} {:>10s} {:>12s}".format(
        "fold", "rows", "row %", "drugs", "proteins"))
    for f in FOLDS:
        d = folds[f]
        print("  {:<6s} {:>8d} {:>7.2f}% {:>10d} {:>12d}".format(
            f, len(d), 100.0 * len(d) / total,
            d[DRUG_COL].nunique(), d[PROT_COL].nunique()))
    print("  {:<6s} {:>8d} {:>7.2f}%".format("total", total, 100.0))
    print("  target row split was 70.00% / 10.00% / 20.00% (drift is expected)")
    print("  positives (Y=1) per fold: " + ", ".join(
        "{}={:.2%}".format(f, (folds[f][LABEL_COL] == 1).mean()) for f in FOLDS))


def write(folds, data, subdir):
    out_dir = os.path.join("datasets", data, subdir)
    os.makedirs(out_dir, exist_ok=True)
    for f in FOLDS:
        path = os.path.join(out_dir, f + ".csv")
        folds[f][COLUMNS].to_csv(path, index=False)
        print("  wrote {:>7d} rows -> {}".format(len(folds[f]), path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, choices=["bindingdb", "biosnap"])
    parser.add_argument("--seed", default=2048, type=int)
    args = parser.parse_args()

    print("=" * 72)
    print("make_splits: {}   (seed {})".format(args.data, args.seed))
    print("=" * 72)

    df = deduplicate(load_random(args.data))

    # Build and validate everything before touching disk, so a failed
    # assertion leaves no half-written split behind.
    cold_drug = cold_split(df, DRUG_COL, args.seed)
    cold_protein = cold_split(df, PROT_COL, args.seed)

    print("\n[validate]")
    check_disjoint(cold_drug, DRUG_COL, "cold_drug")
    check_disjoint(cold_protein, PROT_COL, "cold_protein")

    for folds, name in ((cold_drug, "cold_drug"), (cold_protein, "cold_protein")):
        n = sum(len(folds[f]) for f in FOLDS)
        assert n == len(df), "{}: {} rows out, {} in".format(name, n, len(df))

    print("\n[cold_drug]  drugs held out; proteins unconstrained")
    report(cold_drug, "row-level outcome")
    print("\n[cold_protein]  proteins held out; drugs unconstrained")
    report(cold_protein, "row-level outcome")

    print("\n[write]")
    write(cold_drug, args.data, "cold_drug")
    write(cold_protein, args.data, "cold_protein")
    print()


if __name__ == "__main__":
    main()
