"""Split the cleaned tickets into train / validation / test (about 70 / 15 / 15).

The split is:
  - stratified: each split has roughly the same class mix as the full data;
  - group-aware: every near-duplicate cluster (same group_id) lands entirely
    in ONE split, so the model can't score well on test just by having seen
    a near-copy of a test ticket in training.

How:
  1. Clusters of BIG_CLUSTER_SIZE+ near-identical tickets (recurring templates
     like purchase-order emails) always go to train. In production you'd
     always have history for these, and one 638-ticket template would
     otherwise dominate whichever evaluation split it landed in.
  2. StratifiedGroupKFold cuts the remaining rows into 20 folds of about 5%
     each. Each fold keeps groups whole and tries to match the class mix.
  3. The folds are dealt out at random (seeded): 3 to test, 3 to validation,
     14 to train. Random matters: StratifiedGroupKFold places the biggest
     groups into folds 0, 1, 2... first, so always taking folds 0-2 as test
     would systematically put the biggest clusters in test.

Exact duplicates were already removed in clean_data.py (0 found), so the same
ticket cannot appear on both sides of a split.

Run:  python -m src.split_data
Reads:  data/processed/tickets_clean.csv
Writes: data/processed/{train,val,test}.csv and reports/split_report.md
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import DATA_PROCESSED, RANDOM_SEED, REPORTS_DIR

CLEAN_FILE = DATA_PROCESSED / "tickets_clean.csv"
N_FOLDS = 20
TEST_FOLDS = 3  # 3 / 20 = 15%
VAL_FOLDS = 3   # 3 / 20 = 15%
BIG_CLUSTER_SIZE = 50  # clusters at least this big always go to train


def split_train_val_test(df, n_folds=N_FOLDS, test_folds=TEST_FOLDS, val_folds=VAL_FOLDS,
                         big_cluster_size=BIG_CLUSTER_SIZE, seed=RANDOM_SEED):
    """Return a Series of "train" / "val" / "test", one entry per row of df."""
    split = pd.Series("train", index=df.index, name="split")

    group_size = df["group_id"].map(df["group_id"].value_counts())
    rest = df[group_size < big_cluster_size]  # big clusters stay "train"

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of_row = np.empty(len(rest), dtype=int)
    # Each iteration gives the row positions of one fold as its "held-out" part;
    # we only need those to know which fold every row belongs to.
    for fold, (_, fold_rows) in enumerate(sgkf.split(rest["text"], rest["label"], groups=rest["group_id"])):
        fold_of_row[fold_rows] = fold

    fold_order = np.random.default_rng(seed).permutation(n_folds)
    test_set = fold_order[:test_folds]
    val_set = fold_order[test_folds:test_folds + val_folds]
    split[rest.index[np.isin(fold_of_row, test_set)]] = "test"
    split[rest.index[np.isin(fold_of_row, val_set)]] = "val"
    return split


def main():
    df = pd.read_csv(CLEAN_FILE)
    df["split"] = split_train_val_test(df)

    parts = {name: df[df["split"] == name].drop(columns="split") for name in ["train", "val", "test"]}
    for name, part in parts.items():
        part.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)

    # Sanity checks: these must hold, or the split is broken.
    groups = {name: set(part["group_id"]) for name, part in parts.items()}
    texts = {name: set(part["text"]) for name, part in parts.items()}
    checks = {
        "every row assigned to exactly one split": sum(len(p) for p in parts.values()) == len(df),
        "no group_id shared between splits": not (groups["train"] & groups["val"]
                                                  or groups["train"] & groups["test"]
                                                  or groups["val"] & groups["test"]),
        "no identical text shared between splits": not (texts["train"] & texts["val"]
                                                        or texts["train"] & texts["test"]
                                                        or texts["val"] & texts["test"]),
    }

    group_size = df["group_id"].map(df["group_id"].value_counts())
    big = df[group_size >= BIG_CLUSTER_SIZE]
    out = ["# Split report", "", f"Seed: {RANDOM_SEED}. Clusters of {BIG_CLUSTER_SIZE}+ tickets forced to train: "
           f"{big['group_id'].nunique()} clusters, {len(big):,} rows. Remaining rows: StratifiedGroupKFold, "
           f"{N_FOLDS} folds dealt at random ({N_FOLDS - TEST_FOLDS - VAL_FOLDS} train / {VAL_FOLDS} val / "
           f"{TEST_FOLDS} test).", "",
           "| Split | Rows | % of total | Groups |", "|---|---:|---:|---:|"]
    out += [f"| {n} | {len(p):,} | {len(p) / len(df) * 100:.1f} | {p['group_id'].nunique():,} |"
            for n, p in parts.items()]

    out += ["", "## Class mix per split (% of that split)", "",
            "| Class | All | Train | Val | Test |", "|---|---:|---:|---:|---:|"]
    mix = {n: p["label"].value_counts(normalize=True) * 100 for n, p in parts.items()}
    overall = df["label"].value_counts(normalize=True) * 100
    for c in overall.index:
        out += [f"| {c} | {overall[c]:.2f} | {mix['train'][c]:.2f} | {mix['val'][c]:.2f} | {mix['test'][c]:.2f} |"]

    out += ["", "## Class counts per split", "", "| Class | Train | Val | Test |", "|---|---:|---:|---:|"]
    for c in overall.index:
        out += [f"| {c} | " + " | ".join(f"{(p['label'] == c).sum():,}" for p in parts.values()) + " |"]

    out += ["", "## Where the 5 biggest near-duplicate clusters went", ""]
    sizes = df["group_id"].value_counts().head(5)
    for gid, size in sizes.items():
        rows = df[df["group_id"] == gid]
        out += [f"- group {gid}: {size} tickets, label(s) {sorted(rows['label'].unique())}, "
                f"split: {sorted(rows['split'].unique())}"]

    out += ["", "## Checks", ""] + [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks.items()]
    report = "\n".join(out)
    (REPORTS_DIR / "split_report.md").write_text(report, encoding="utf-8")
    print(report)
    if not all(checks.values()):
        raise SystemExit("Split checks failed; see above.")


if __name__ == "__main__":
    main()
