"""Probe the final model with hand-written tickets, and check how the training
data labels a few telling words. This backs up the README's limitations.

These hand-written tickets are NOT an evaluation (that's the test set). They're
spot checks that show behaviour a user of the app will quickly run into.

Run:  python -m src.probe_examples
Writes: reports/probe_examples.md
"""
import numpy as np
import pandas as pd

from src.config import REPORTS_DIR
from src.evaluate import load_split
from src.predict import load_model, predict_ticket

PROBE_TICKETS = [
    "Hi, my laptop will not connect to the docking station monitors since this morning.",
    "I forgot my password and am locked out",
    "please add me to the shared drive folder",
    "New starter joining Monday, please set up onboarding.",
    "Please grant me admin rights on my machine so I can install Visual Studio.",
    "We need to order 5 new keyboards and a monitor for the new team, PO attached.",
    "google app is frozen",
    "my google app is frozen and wont open",
    "the google app is frozen on my laptop",
    "Google Chrome keeps freezing and crashing",
    "outlook is not opening",
    "zzz",
]
PROBE_WORDS = ["frozen", "freezing", "crashing", "chrome", "software", "install", "outlook", "google"]


def main():
    model = load_model()
    vocab = model.named_steps["tfidf"].vocabulary_
    analyzer = model.named_steps["tfidf"].build_analyzer()

    out = ["# Probe examples (hand-written tickets, not an evaluation)", "",
           "| Ticket | Top 3 (confidence) | Words/phrases the model knows |", "|---|---|---|"]
    for t in PROBE_TICKETS:
        top = predict_ticket(t, model)["top"]
        known = [w for w in analyzer(t) if w in vocab and " " not in w]
        out += [f"| {t} | " + ", ".join(f"{label} {p:.2f}" for label, p in top) + f" | {', '.join(known) or '(none)'} |"]

    texts, labels, _ = load_split("train")
    out += ["", "## How the training data labels some words", "",
            "| Word | Training tickets containing it | Top labels (% of those tickets) |", "|---|---:|---|"]
    for w in PROBE_WORDS:
        mask = texts.str.contains(rf"\b{w}\b")
        mix = labels[mask].value_counts(normalize=True).head(3) * 100
        out += [f"| {w} | {mask.sum():,} | " + (", ".join(f"{c} {v:.0f}%" for c, v in mix.items()) or "-") + " |"]

    clf = model.named_steps["clf"]
    out += ["", "## What the model predicts when it knows no words at all", "",
            f"With an all-zero feature vector only the intercepts count, and the largest is "
            f"**{clf.classes_[np.argmax(clf.intercept_)]}**: "
            + ", ".join(f"{c} {v:.2f}" for c, v in sorted(zip(clf.classes_, clf.intercept_), key=lambda x: -x[1])), ""]

    report = "\n".join(out)
    (REPORTS_DIR / "probe_examples.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
