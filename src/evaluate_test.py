"""Stage 6: the ONE evaluation on the held-out test set.

  1. Pick the final model: the highest validation macro F1 in
     reports/results_validation.csv (decided in Stage 5, before any test data is seen).
  2. Score it on test, along with the two Stage 4 baselines for comparison.
     The baselines' test scores are reported only; they don't influence any choice.
  3. Save the confusion matrix, the per-class report, and a seeded random
     sample of misclassified tickets.
  4. Save the final pipeline to models/ticket_classifier.joblib for the app.

The model is used exactly as tuned and fit on train. It is not refit on
train + val, so the deployed model is the same object that got the test score.

This script refuses to run twice (so the test set isn't reused by accident).
Pass --rerun to reproduce the same numbers on purpose (the result is
deterministic, so a rerun gives identical output).

Run:  python -m src.evaluate_test
"""
import shutil
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_SEED, REPORTS_DIR
from src.evaluate import RESULTS_FILE, load_split, per_class_report, score, to_markdown
from src.predict import MODEL_FILE
from src.train_models import CANDIDATES_DIR, slug

TEST_REPORT = REPORTS_DIR / "test_results.md"
BASELINES = ["Majority class", "TF-IDF + LogReg (default)"]
N_EXAMPLES = 15


def plot_confusion(y_true, y_pred, labels, path):
    """Two panels: raw counts, and rows normalised to recall (each row sums to 1)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    for ax, norm, title in [(axes[0], None, "Counts"), (axes[1], "true", "Share of each true class (row = recall)")]:
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=labels, normalize=norm, ax=ax, cmap="Blues",
            colorbar=False, xticks_rotation=45, values_format="d" if norm is None else ".2f")
        ax.set_title(title)
    fig.suptitle("Final model on the test set")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    if TEST_REPORT.exists() and "--rerun" not in sys.argv:
        sys.exit(f"{TEST_REPORT.name} already exists: the test set has been used. "
                 "Pass --rerun only to reproduce the same numbers.")

    val_table = pd.read_csv(RESULTS_FILE).sort_values("macro_f1", ascending=False)
    final_name = val_table.iloc[0]["model"]
    print(f"Final model (best validation macro F1): {final_name}")

    X_test, y_test, _ = load_split("test")
    labels = list(y_test.value_counts().index)

    rows, final_pred = [], None
    for name in [final_name] + BASELINES:
        model = joblib.load(CANDIDATES_DIR / f"{slug(name)}.joblib")
        pred = model.predict(X_test)
        rows.append({"model": name, **score(y_test, pred)})
        if name == final_name:
            final_model, final_pred = model, pred
    test_table = pd.DataFrame(rows)

    plot_confusion(y_test, final_pred, labels, FIGURES_DIR / "confusion_matrix_test.png")
    cm = pd.DataFrame(confusion_matrix(y_test, final_pred, labels=labels), index=labels, columns=labels)

    # Misclassified examples: a seeded random sample, so it isn't cherry-picked.
    probs = final_model.predict_proba(X_test)
    classes = list(final_model.classes_)
    test = pd.DataFrame({"text": X_test, "true": y_test, "pred": final_pred})
    test["confidence"] = probs.max(axis=1)
    test["prob_of_true"] = [probs[i, classes.index(t)] for i, t in enumerate(y_test)]
    wrong = test[test["true"] != test["pred"]]
    sample = wrong.sample(n=N_EXAMPLES, random_state=RANDOM_SEED)
    wrong.to_csv(REPORTS_DIR / "test_misclassified_all.csv", index_label="test_row")

    # Most common confusions (true -> predicted), from the counts matrix.
    pairs = cm.stack().rename_axis(["true", "predicted"]).reset_index(name="count")
    pairs = pairs[pairs["true"] != pairs["predicted"]].sort_values("count", ascending=False).head(8)

    out = ["# Test set results (single evaluation)", "",
           f"Final model: **{final_name}**, chosen by validation macro F1 "
           f"({val_table.iloc[0]['macro_f1']:.4f}) before the test set was used.",
           f"Test rows: {len(X_test):,}. Baselines are shown for comparison only.", "",
           "## Overall", "", to_markdown(test_table), "",
           "## Validation vs test (final model)", "",
           to_markdown(pd.DataFrame([
               {"set": "validation", **val_table.iloc[0][["accuracy", "macro_f1", "weighted_f1"]].to_dict()},
               {"set": "test", **test_table.iloc[0][["accuracy", "macro_f1", "weighted_f1"]].to_dict()},
           ])), "",
           "## Per-class (final model)", "",
           to_markdown(per_class_report(y_test, final_pred).reset_index(names="class")), "",
           "## Confusion matrix (counts; rows = true class, columns = predicted)", "",
           to_markdown(cm.reset_index(names="true \\ predicted")), "",
           "![confusion matrix](figures/confusion_matrix_test.png)", "",
           "## Most common confusions", "", to_markdown(pairs), "",
           f"## {N_EXAMPLES} misclassified examples (random sample, seed {RANDOM_SEED}, "
           f"out of {len(wrong):,} errors)", ""]
    for n, (i, r) in enumerate(sample.iterrows(), start=1):
        out += [f"**{n}. test row {i}** | true: **{r['true']}** | predicted: **{r['pred']}** "
                f"(confidence {r['confidence']:.2f}; probability given to true class {r['prob_of_true']:.2f})", "",
                f"> {r['text'][:400]}{'...' if len(r['text']) > 400 else ''}", ""]

    report = "\n".join(out)
    TEST_REPORT.write_text(report, encoding="utf-8")
    test_table.to_csv(REPORTS_DIR / "results_test.csv", index=False, float_format="%.4f")
    shutil.copyfile(CANDIDATES_DIR / f"{slug(final_name)}.joblib", MODEL_FILE)
    print(report)
    print(f"\nSaved final pipeline to {MODEL_FILE.relative_to(MODELS_DIR.parent)}")


if __name__ == "__main__":
    main()
