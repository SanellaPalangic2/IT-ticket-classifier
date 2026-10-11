"""Load the final model and classify one ticket. Used by the Streamlit app.

    from src.predict import load_model, predict_ticket
    model = load_model()
    predict_ticket("cannot log in to vpn after password reset", model)
    -> {"label": "Access", "top": [("Access", 0.81), ("Hardware", 0.07), ...]}
"""
import joblib
import pandas as pd

from src.clean_data import mask_names, normalize_whitespace
from src.config import MODELS_DIR

MODEL_FILE = MODELS_DIR / "ticket_classifier.joblib"


def load_model(path=MODEL_FILE):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.evaluate_test` first to create it.")
    return joblib.load(path)


def predict_ticket(text, model, top_k=3):
    """Return the predicted category and the top_k classes with probabilities.

    New text gets the same cleaning as the training data (whitespace, name
    masking). The Pipeline then applies the TF-IDF it learned from training
    text. Empty or whitespace-only input returns label None instead of
    crashing, because an empty ticket has nothing to classify.
    """
    cleaned = mask_names(normalize_whitespace(pd.Series([text]))).iloc[0]
    if cleaned == "":
        return {"label": None, "top": []}

    probs = model.predict_proba([cleaned])[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda pair: pair[1], reverse=True)[:top_k]
    return {"label": ranked[0][0], "top": [(label, float(p)) for label, p in ranked]}
