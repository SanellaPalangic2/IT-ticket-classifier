"""Tests for the cleaning functions, using tiny hand-made examples."""
import pandas as pd

from src.clean_data import (
    drop_duplicates,
    drop_empty,
    mask_names,
    merge_rare_classes,
    normalize_whitespace,
)
from src.dedup import assign_groups


def test_normalize_whitespace_collapses_and_strips():
    s = pd.Series(["  printer   broken \n", None, "ok"])
    assert normalize_whitespace(s).tolist() == ["printer broken", "", "ok"]


def test_drop_empty_removes_only_empty_text():
    df = pd.DataFrame({"text": ["vpn down", "", "reset password"], "label": ["A", "A", "B"]})
    assert drop_empty(df)["text"].tolist() == ["vpn down", "reset password"]


def test_drop_duplicates_keeps_one_copy():
    df = pd.DataFrame({"text": ["vpn down", "vpn down", "new laptop"], "label": ["A", "A", "B"]})
    assert drop_duplicates(df)["text"].tolist() == ["vpn down", "new laptop"]


def test_drop_duplicates_removes_conflicting_labels_entirely():
    df = pd.DataFrame({"text": ["vpn down", "vpn down", "new laptop"], "label": ["A", "B", "B"]})
    assert drop_duplicates(df)["text"].tolist() == ["new laptop"]


def test_mask_names_matches_whole_words_only():
    s = pd.Series(["admin rights for florin", "permission version ion", "florins"])
    assert mask_names(s).tolist() == [
        "admin rights for <name>",
        "permission version <name>",
        "florins",
    ]


def test_merge_rare_classes():
    labels = pd.Series(["A"] * 5 + ["B"] * 2)
    assert merge_rare_classes(labels, min_size=3).tolist() == ["A"] * 5 + ["Other"] * 2


def test_assign_groups_puts_near_duplicates_together():
    texts = [
        "leaver form leave date location tower hello leaver form filled",
        "leaver form leave date location tower hello leaver form filled please",
        "monitor flickering after docking station update",
    ]
    groups = assign_groups(texts)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
