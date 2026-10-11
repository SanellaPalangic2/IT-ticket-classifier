"""Tests for the train / validation / test split on a small synthetic frame."""
import pandas as pd

from src.split_data import split_train_val_test


def make_frame():
    # 200 rows, 2 classes, groups of 4 rows that share a label.
    rows = []
    for g in range(50):
        label = "A" if g % 2 == 0 else "B"
        rows += [{"text": f"ticket {g} copy {k}", "label": label, "group_id": g} for k in range(4)]
    return pd.DataFrame(rows)


def test_every_row_gets_one_split():
    df = make_frame()
    split = split_train_val_test(df, n_folds=10, test_folds=2, val_folds=2)
    assert set(split) == {"train", "val", "test"}
    assert len(split) == len(df)


def test_groups_never_cross_splits():
    df = make_frame()
    df["split"] = split_train_val_test(df, n_folds=10, test_folds=2, val_folds=2)
    assert (df.groupby("group_id")["split"].nunique() == 1).all()


def test_big_clusters_go_to_train():
    df = make_frame()
    big = pd.DataFrame({"text": [f"template copy {k}" for k in range(30)], "label": "A", "group_id": 999})
    df = pd.concat([df, big], ignore_index=True)
    df["split"] = split_train_val_test(df, n_folds=10, test_folds=2, val_folds=2, big_cluster_size=10)
    assert (df.loc[df["group_id"] == 999, "split"] == "train").all()


def test_split_is_reproducible():
    df = make_frame()
    a = split_train_val_test(df, n_folds=10, test_folds=2, val_folds=2)
    b = split_train_val_test(df, n_folds=10, test_folds=2, val_folds=2)
    assert a.equals(b)
