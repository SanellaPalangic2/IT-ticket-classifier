"""Shared helpers for scoring a model and recording results.

Every model (baseline or stronger) is scored with the same function, so the
numbers in the results table are directly comparable.
"""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from src.config import DATA_PROCESSED, REPORTS_DIR

RESULTS_FILE = REPORTS_DIR / "results_validation.csv"


def load_split(name):
    """Load train / val / test as (texts, labels, group_ids)."""
    df = pd.read_csv(DATA_PROCESSED / f"{name}.csv")
    return df["text"], df["label"], df["group_id"]


def score(y_true, y_pred):
    """Overall metrics: accuracy plus macro and weighted precision/recall/F1.

    zero_division=0: if a model never predicts some class (the majority
    baseline never predicts 7 of the 8), that class's precision is undefined
    (0 / 0). We count it as 0 instead of letting scikit-learn warn.
    """
    metrics = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ["macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        metrics.update({f"{avg}_precision": p, f"{avg}_recall": r, f"{avg}_f1": f})
    return metrics


def per_class_report(y_true, y_pred):
    """Per-class precision / recall / F1 / support as a DataFrame."""
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = {k: v for k, v in rep.items() if k not in ("accuracy", "macro avg", "weighted avg")}
    df = pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]]
    df["support"] = df["support"].astype(int)
    return df.sort_values("support", ascending=False)


def record_result(model_name, metrics, path=RESULTS_FILE):
    """Add or replace one model's row in the shared validation results table."""
    row = pd.DataFrame([{"model": model_name, **metrics}])
    if path.exists():
        table = pd.read_csv(path)
        table = pd.concat([table[table["model"] != model_name], row], ignore_index=True)
    else:
        table = row
    table.to_csv(path, index=False, float_format="%.4f")
    return table


def to_markdown(df, floatfmt=".4f"):
    """Small markdown table writer, so we don't need the extra 'tabulate' package."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
