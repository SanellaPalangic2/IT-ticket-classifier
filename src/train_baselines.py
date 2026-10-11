"""Stage 4: two baselines, scored on the validation set.

  (a) Majority class: always predicts the most common training label.
      Any real model must beat this, or it has learned nothing.
  (b) TF-IDF + logistic regression with default settings, in one Pipeline.

The test set is NOT loaded here. It gets used once, in Stage 6.

Run:  python -m src.train_baselines
Writes: reports/results_validation.csv (shared results table)
        reports/baselines_validation.md (overall and per-class results)
"""
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_SEED, REPORTS_DIR
from src.evaluate import load_split, per_class_report, record_result, score, to_markdown


def build_majority_baseline():
    return DummyClassifier(strategy="most_frequent")


def build_tfidf_logreg():
    """TF-IDF features feeding a logistic regression, as ONE Pipeline object.

    Why a Pipeline: calling pipeline.fit(X_train, y_train) fits the
    vectorizer's vocabulary and IDF weights on training text only. At
    validation and test time, pipeline.predict() only *transforms* that text
    with what was learned from train. If we fit TF-IDF on all the data first,
    words and word frequencies from validation/test tickets would leak into
    the features, and the scores would be optimistic. Inside
    cross-validation, the Pipeline also refits the vectorizer on each training
    fold automatically.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer()),  # defaults: lowercase, single words, L2-normalised rows
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])


def main():
    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")

    models = {
        "Majority class": build_majority_baseline(),
        "TF-IDF + LogReg (default)": build_tfidf_logreg(),
    }

    out = ["# Baselines (validation set)", "",
           f"Train rows: {len(X_train):,}. Validation rows: {len(X_val):,}. Test set not used.", ""]
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        metrics = score(y_val, pred)
        table = record_result(name, metrics)
        out += [f"## {name}", "", "Per-class results:", "",
                to_markdown(per_class_report(y_val, pred).reset_index(names="class")), ""]

    out = out[:4] + ["## Overall", "", to_markdown(table), ""] + out[4:]
    report = "\n".join(out)
    (REPORTS_DIR / "baselines_validation.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
