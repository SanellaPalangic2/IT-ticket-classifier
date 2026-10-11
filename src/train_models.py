"""Stage 5: tune stronger models with cross-validation on the TRAINING set only,
then score each tuned model once on validation.

Candidates (all CPU-friendly linear models on sparse TF-IDF features):
  - the two Stage 4 baselines (no tuning; included so they get a CV score too)
  - TF-IDF + logistic regression, tuned (C, word n-gram range, class weights)
  - TF-IDF + LinearSVC, tuned (same knobs)
  - word + character n-grams + LinearSVC, tuned (C)

Cross-validation is StratifiedGroupKFold on train.csv using group_id, so
near-duplicate tickets never sit on both sides of a CV fold (same rule as
the main split). The score being optimised is macro F1.

The test set is NOT loaded here.

Run:  python -m src.train_models
Writes: models/candidates/<name>.joblib    (each tuned pipeline, fit on all of train)
        reports/results_validation.csv     (shared results table, adds cv columns)
        reports/model_selection.md         (grids, best settings, results table)
"""
import time

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.config import MODELS_DIR, RANDOM_SEED, REPORTS_DIR
from src.evaluate import RESULTS_FILE, load_split, per_class_report, record_result, score, to_markdown

CANDIDATES_DIR = MODELS_DIR / "candidates"
CV_FOLDS = 5


def word_tfidf():
    # sublinear_tf: use 1 + log(count) instead of the raw count, so a word
    # repeated 20 times in a long pasted email thread doesn't dominate.
    return TfidfVectorizer(sublinear_tf=True)


def build_candidates():
    """Return {name: (pipeline, param_grid)}. An empty grid means 'no tuning'."""
    return {
        "Majority class": (DummyClassifier(strategy="most_frequent"), {}),
        "TF-IDF + LogReg (default)": (
            Pipeline([("tfidf", TfidfVectorizer()),
                      ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))]),
            {},
        ),
        "TF-IDF + LogReg (tuned)": (
            Pipeline([("tfidf", word_tfidf()),
                      ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))]),
            {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__C": [1, 5, 20],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "TF-IDF + LinearSVC (tuned)": (
            Pipeline([("tfidf", word_tfidf()),
                      ("clf", LinearSVC(random_state=RANDOM_SEED))]),
            {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__C": [0.1, 0.3, 1],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "Word + char n-grams + LinearSVC (tuned)": (
            Pipeline([
                ("features", FeatureUnion([
                    ("word", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))),
                    # char_wb: character n-grams taken only inside word boundaries.
                    # They catch word pieces and typos ("pasword", "printr") that
                    # whole-word features miss. min_df=3 drops very rare n-grams to
                    # keep the feature count manageable on a laptop.
                    ("char", TfidfVectorizer(sublinear_tf=True, analyzer="char_wb",
                                             ngram_range=(2, 5), min_df=3)),
                ])),
                ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_SEED)),
            ]),
            {"clf__C": [0.1, 0.3, 1]},
        ),
    }


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def main():
    X_train, y_train, g_train = load_split("train")
    X_val, y_val, _ = load_split("val")
    CANDIDATES_DIR.mkdir(exist_ok=True)

    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    out = ["# Model selection", "",
           f"Tuning: GridSearchCV, {CV_FOLDS}-fold StratifiedGroupKFold on train ({len(X_train):,} rows), "
           f"scoring = macro F1, seed = {RANDOM_SEED}. Each tuned model is then refit on all of train "
           f"and scored once on validation ({len(X_val):,} rows). Test set not used.", ""]
    per_class = []

    for name, (pipeline, grid) in build_candidates().items():
        start = time.time()
        search = GridSearchCV(pipeline, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
        # groups= is passed through to StratifiedGroupKFold so CV folds respect clusters.
        search.fit(X_train, y_train, groups=g_train)
        minutes = (time.time() - start) / 60

        best = search.best_estimator_
        pred = best.predict(X_val)
        i = search.best_index_
        metrics = {
            "cv_macro_f1": search.cv_results_["mean_test_score"][i],
            "cv_macro_f1_std": search.cv_results_["std_test_score"][i],
            **score(y_val, pred),
        }
        record_result(name, metrics)
        joblib.dump(best, CANDIDATES_DIR / f"{slug(name)}.joblib")

        n_combos = len(search.cv_results_["params"])
        out += [f"## {name}", "",
                f"- Settings tried: {n_combos} x {CV_FOLDS} folds; time {minutes:.1f} min",
                f"- Grid: {grid or 'none (fixed settings)'}",
                f"- Best settings: {search.best_params_ or 'n/a'}",
                f"- CV macro F1: {metrics['cv_macro_f1']:.4f} (+/- {metrics['cv_macro_f1_std']:.4f})",
                f"- Validation macro F1: {metrics['macro_f1']:.4f}", ""]
        per_class.append((name, per_class_report(y_val, pred)))
        print(f"done: {name} ({minutes:.1f} min) val macro F1 = {metrics['macro_f1']:.4f}", flush=True)

    table =pd.read_csv(RESULTS_FILE).sort_values("macro_f1", ascending=False)
    out[3:3] = ["## Results (sorted by validation macro F1)", "", to_markdown(table), ""]

    out += ["## Per-class F1 on validation", ""]
    f1 = pd.DataFrame({name: rep["f1-score"] for name, rep in per_class if "Majority" not in name})
    f1.insert(0, "support", per_class[0][1]["support"])
    out += [to_markdown(f1.reset_index(names="class")), ""]

    report = "\n".join(out)
    (REPORTS_DIR / "model_selection.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
