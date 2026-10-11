"""Tests for the prediction function.

Most tests use a tiny model trained right here, so they run without the
real (git-ignored) model file. The last test uses the real saved model and
is skipped if it hasn't been created yet.
"""
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.predict import MODEL_FILE, load_model, predict_ticket


@pytest.fixture(scope="module")
def tiny_model():
    texts = ["printer jammed paper", "monitor broken screen", "laptop keyboard broken",
             "reset password account locked", "need access shared folder", "grant access vpn account",
             "new starter onboarding form", "leaver form leave date", "holiday request hr"]
    labels = ["Hardware"] * 3 + ["Access"] * 3 + ["HR Support"] * 3
    return Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())]).fit(texts, labels)


def test_normal_input_returns_a_valid_class(tiny_model):
    result = predict_ticket("my printer is jammed again", tiny_model)
    assert result["label"] in tiny_model.classes_
    assert result["label"] == "Hardware"


def test_top_k_is_sorted_and_probabilities_are_valid(tiny_model):
    top = predict_ticket("locked out of my account", tiny_model)["top"]
    assert len(top) == 3
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True)
    assert all(0 <= p <= 1 for p in probs)


@pytest.mark.parametrize("empty", ["", "   ", "\n\t", None])
def test_empty_input_does_not_crash(tiny_model, empty):
    assert predict_ticket(empty, tiny_model) == {"label": None, "top": []}


def test_unknown_words_still_return_a_class(tiny_model):
    # No word here was seen in training: the model must still answer, not crash.
    assert predict_ticket("zzzz qqqq", tiny_model)["label"] in tiny_model.classes_


@pytest.mark.skipif(not MODEL_FILE.exists(), reason="final model not trained yet")
def test_real_model_predicts_valid_class():
    model = load_model()
    result = predict_ticket("Hi, I can't log in to the VPN since my password expired. Thanks!", model)
    assert result["label"] in model.classes_
    assert len(model.classes_) == 8
