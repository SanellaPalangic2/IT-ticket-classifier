"""Tests for the shared scoring helpers."""
import pytest

from src.evaluate import record_result, score


def test_score_perfect_predictions():
    m = score(["A", "B", "B"], ["A", "B", "B"])
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0


def test_score_macro_vs_weighted_on_majority_guess():
    # 3 x A, 1 x B; always guess A. A has F1 = 0.857, B has F1 = 0.
    m = score(["A", "A", "A", "B"], ["A", "A", "A", "A"])
    assert m["accuracy"] == 0.75
    assert m["macro_f1"] == pytest.approx((6 / 7 + 0) / 2)        # every class counts equally
    assert m["weighted_f1"] == pytest.approx((6 / 7) * 3 / 4)     # weighted by class size


def test_record_result_replaces_existing_row(tmp_path):
    path = tmp_path / "results.csv"
    record_result("m1", {"macro_f1": 0.5}, path)
    record_result("m2", {"macro_f1": 0.6}, path)
    table = record_result("m1", {"macro_f1": 0.7}, path)
    assert sorted(table["model"]) == ["m1", "m2"]
    assert table.loc[table["model"] == "m1", "macro_f1"].item() == 0.7
