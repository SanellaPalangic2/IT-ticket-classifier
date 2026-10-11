"""Streamlit demo: type an IT ticket, get the predicted category.

Run (from the project folder, with the venv active):
    streamlit run app.py
Then open http://localhost:8501 in your browser.

Needs models/ticket_classifier.joblib, which `python -m src.evaluate_test` creates.
"""
import pandas as pd
import streamlit as st

from src.config import REPORTS_DIR
from src.predict import load_model, predict_ticket

RESULTS_TEST = REPORTS_DIR / "results_test.csv"

st.set_page_config(page_title="IT Ticket Classifier", page_icon="🎫")


# cache_resource: load the model once per server, not on every click.
# Streamlit reruns this whole script top to bottom on each interaction.
@st.cache_resource
def get_model():
    return load_model()


st.title("IT Ticket Classifier")
st.write("Type a help-desk ticket and the model suggests which category it should be routed to.")

try:
    model = get_model()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

# A ?ticket=... in the URL pre-fills the box (handy for demos and screenshots).
default_text = st.query_params.get("ticket", "")
text = st.text_area("Ticket text", value=default_text, height=150,
                    placeholder="e.g. Hi, my laptop won't connect to the docking station monitors since this morning.")

if st.button("Classify", type="primary") or default_text:
    result = predict_ticket(text, model)
    if result["label"] is None:
        st.warning("Please enter some ticket text first.")
    else:
        st.subheader(f"Predicted category: {result['label']}")
        top = pd.DataFrame(result["top"], columns=["Category", "Confidence"])
        st.bar_chart(top, x="Category", y="Confidence", horizontal=True, sort=False)
        top["Confidence"] = top["Confidence"].map("{:.1%}".format)
        st.table(top.set_index("Category"))
        st.caption("Confidence is the model's probability score. It hasn't been checked for "
                   "calibration, so treat it as a ranking, not a guarantee.")

with st.expander("About this model"):
    # Scores are read from the file the test evaluation wrote, never typed in by hand.
    scores = ""
    if RESULTS_TEST.exists():
        final = pd.read_csv(RESULTS_TEST).iloc[0]  # first row = the final model
        scores = f" Test-set macro F1: {final['macro_f1']:.4f}; accuracy: {final['accuracy']:.4f}."
    st.write(
        "TF-IDF (words and word pairs) + logistic regression with balanced class weights, "
        f"trained on a public Kaggle dataset of IT tickets in {len(model.classes_)} categories.{scores} "
        "The training text was lowercased with stopwords removed, so very short or unusual "
        "tickets may be misrouted."
    )
