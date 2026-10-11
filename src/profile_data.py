"""Profile the raw ticket dataset before any cleaning.

Answers: how big is it, how imbalanced are the classes, how long are the
tickets, is any text missing, are there duplicates, could any column leak
the label, and is there personal information in the text.

Run:  python -m src.profile_data
Writes: reports/data_profile.md and two figures in reports/figures/.
It reads the raw data and changes nothing.
"""
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # draw to files only; no window pops up
import matplotlib.pyplot as plt
import pandas as pd

from src.config import DATA_RAW, FIGURES_DIR, REPORTS_DIR
from src.dedup import NEAR_DUP_THRESHOLD, find_near_duplicates, groups_from_pairs

RAW_FILE = DATA_RAW / "all_tickets_processed_improved_v3.csv"
TEXT_COL = "Document"
LABEL_COL = "Topic_group"

# Patterns for personal information. They are deliberately broad, so they
# over-count rather than miss things, and every hit gets inspected by hand.
PII_PATTERNS = {
    "email address": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "email-like 'at' token pair": r"\b\w+ at \w+ (?:com|net|org)\b",
    "phone-like digit run (7+ digits)": r"(?:\d[\s-]?){7,}",
    "any digit": r"\d",
    "url": r"https?://|www\.",
    "ip address": r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
}
# Names usually appear right after a greeting or before a sign-off, so we
# look at which words sit in those positions.
NAME_CONTEXT = r"\b(?:dear|hi|hello|regards|thanks|thank you|mr|mrs|ms)\s+(\w+)"
# A short list of common English and Romanian first names (the text mentions
# Bucharest, so Romanian names are likely). A spot check, not a complete scan.
COMMON_FIRST_NAMES = """james john robert michael william david richard joseph thomas charles
mary patricia jennifer linda elizabeth barbara susan jessica sarah karen daniel paul mark
george andrew peter anna maria elena ana ion andrei alexandru mihai cristina ioana alex
adrian bogdan florin radu gabriel laura andreea diana raluca simona irina victor sergiu
vlad dan ben tom sam chris emma olivia sophie lucy kate jack harry oliver""".split()


def main():
    df = pd.read_csv(RAW_FILE)
    out = ["# Data profile (raw data)", "", f"Source file: `{RAW_FILE.name}`", ""]

    # --- Shape and columns ---
    out += ["## Shape", f"- Rows: {len(df):,}", f"- Columns: {list(df.columns)}", ""]

    # --- Missing / empty text ---
    text = df[TEXT_COL]
    n_null = text.isna().sum()
    n_blank = (text.fillna("").str.strip() == "").sum()  # includes nulls
    n_null_label = df[LABEL_COL].isna().sum()
    out += ["## Missing or empty values",
            f"- Null text: {n_null}",
            f"- Null or whitespace-only text: {n_blank}",
            f"- Null label: {n_null_label}", ""]

    # --- Class distribution ---
    counts = df[LABEL_COL].value_counts()
    pct = (counts / len(df) * 100).round(2)
    out += ["## Class distribution", "| Class | Count | % |", "|---|---:|---:|"]
    out += [f"| {c} | {counts[c]:,} | {pct[c]} |" for c in counts.index]
    out += ["", f"- Number of classes: {len(counts)}",
            f"- Largest / smallest class ratio: {counts.max() / counts.min():.1f}x",
            f"- Majority-class share (what always guessing '{counts.index[0]}' scores as accuracy): {pct.iloc[0]}%", ""]

    fig, ax = plt.subplots(figsize=(8, 4))
    counts.sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_xlabel("Tickets")
    ax.set_title("Class distribution (raw data)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution_raw.png", dpi=120)
    plt.close(fig)

    # --- Text length ---
    words = text.fillna("").str.split().str.len()
    chars = text.fillna("").str.len()
    q = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    out += ["## Text length", "| Quantile | Words | Characters |", "|---|---:|---:|"]
    out += [f"| {int(p * 100)}% | {words.quantile(p):.0f} | {chars.quantile(p):.0f} |" for p in q]
    out += ["", f"- Tickets with 1-2 words: {(words.between(1, 2)).sum()}",
            f"- Tickets with 3-5 words: {(words.between(3, 5)).sum()}", ""]
    out += ["### Median words per ticket by class", "| Class | Median words |", "|---|---:|"]
    med = words.groupby(df[LABEL_COL]).median().sort_values()
    out += [f"| {c} | {m:.0f} |" for c, m in med.items()]
    out += [""]

    fig, ax = plt.subplots(figsize=(8, 4))
    words.clip(upper=words.quantile(0.99)).plot.hist(bins=60, ax=ax, color="#4C72B0")
    ax.set_xlabel("Words per ticket (clipped at 99th percentile)")
    ax.set_title("Text length distribution (raw data)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "text_length_raw.png", dpi=120)
    plt.close(fig)

    # --- Exact duplicates ---
    valid = df[text.fillna("").str.strip() != ""].copy()
    valid["norm"] = valid[TEXT_COL].str.strip().str.split().str.join(" ")  # collapse whitespace
    dup_rows = valid.duplicated(subset=["norm", LABEL_COL], keep="first").sum()
    labels_per_text = valid.groupby("norm")[LABEL_COL].nunique()
    conflict_texts = labels_per_text[labels_per_text > 1]
    conflict_rows = valid["norm"].isin(conflict_texts.index).sum()
    out += ["## Exact duplicates (after collapsing whitespace)",
            f"- Extra copies of an identical text with the same label: {dup_rows}",
            f"- Distinct texts that appear with MORE THAN ONE label: {len(conflict_texts)} "
            f"(covering {conflict_rows} rows)", ""]
    if len(conflict_texts):
        out += ["Examples of conflicting labels:", ""]
        for t in conflict_texts.index[:5]:
            labs = sorted(valid.loc[valid["norm"] == t, LABEL_COL].unique())
            out += [f"- `{t[:100]}` -> {labs}"]
        out += [""]

    # --- Near duplicates (on unique texts only, so exact copies don't count) ---
    unique = valid.drop_duplicates(subset="norm").reset_index(drop=True)
    pairs = find_near_duplicates(unique["norm"].tolist())
    in_pair = {i for i, _, _ in pairs} | {j for _, j, _ in pairs}
    same_label = sum(unique.at[i, LABEL_COL] == unique.at[j, LABEL_COL] for i, j, _ in pairs)
    out += [f"## Near duplicates (TF-IDF cosine >= {NEAR_DUP_THRESHOLD}, among unique texts)",
            f"- Near-duplicate pairs: {len(pairs):,}",
            f"- Unique texts involved in at least one pair: {len(in_pair):,} of {len(unique):,}",
            f"- Pairs where both texts have the same label: {same_label:,} "
            f"({same_label / max(len(pairs), 1) * 100:.1f}%)", ""]
    # Chain pairs into clusters: if A~B and B~C, then A, B, C form one cluster.
    unique["cluster"] = groups_from_pairs(pairs, len(unique))
    sizes = unique["cluster"].value_counts()
    sizes = sizes[sizes > 1]
    out += [f"- Near-duplicate clusters (2+ texts chained together): {len(sizes):,}",
            f"- Largest 5 cluster sizes: {sizes.head(5).tolist()}"]
    for cid in sizes.index[:3]:
        labs = unique.loc[unique["cluster"] == cid, LABEL_COL].value_counts().to_dict()
        out += [f"  - cluster of {sizes[cid]}: labels {labs}"]
    out += [""]
    out += ["Examples:", ""]
    for i, j, v in sorted(pairs, key=lambda p: p[2])[:: max(len(pairs) // 5, 1)][:5]:
        out += [f"- sim {v:.2f}: `{unique.at[i, 'norm'][:80]}` ({unique.at[i, LABEL_COL]}) vs "
                f"`{unique.at[j, 'norm'][:80]}` ({unique.at[j, LABEL_COL]})"]
    out += [""]

    # --- Label leakage ---
    out += ["## Possible label leakage",
            f"- Columns other than text and label: {[c for c in df.columns if c not in (TEXT_COL, LABEL_COL)] or 'none'}",
            "- Share of tickets whose text contains their own class name as a word:", ""]
    for c in counts.index:
        word = c.lower()
        sub = valid[valid[LABEL_COL] == c]
        hit = sub["norm"].str.contains(rf"\b{re.escape(word)}\b").mean() * 100
        out += [f"  - {c}: {hit:.1f}%"]
    out += [""]

    # --- Personal information ---
    out += ["## Personal information scan", "| Pattern | Tickets matching | Example matches |", "|---|---:|---|"]
    for name, pat in PII_PATTERNS.items():
        matches = valid["norm"].str.findall(pat)
        n = (matches.str.len() > 0).sum()
        examples = Counter(m.strip() for ms in matches for m in ms).most_common(5)
        out += [f"| {name} | {n} | {', '.join(repr(e) for e, _ in examples) or '-'} |"]
    after = Counter(m for ms in valid["norm"].str.findall(NAME_CONTEXT) for m in ms)
    out += ["", "Most common words right after a greeting or sign-off (where names usually appear):", "",
            ", ".join(f"{w} ({n})" for w, n in after.most_common(40)), ""]
    word_counts = Counter(w for t in valid["norm"] for w in t.split())
    found = {n: word_counts[n] for n in COMMON_FIRST_NAMES if word_counts[n]}
    n_tickets = valid["norm"].str.contains(rf"\b(?:{'|'.join(found) or 'x^'})\b").sum()
    out += [f"Common first names found as words (checked {len(COMMON_FIRST_NAMES)} names): "
            f"{found or 'none'}, in {n_tickets} tickets", ""]

    report = "\n".join(out)
    (REPORTS_DIR / "data_profile.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
