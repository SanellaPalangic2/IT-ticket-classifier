"""Clean the raw ticket data and save it for splitting.

Rules, applied in this order:
  1. Normalise whitespace (strip ends, collapse runs of spaces).
  2. Drop tickets whose text is empty after step 1.
  3. Drop exact duplicate texts. If one text appears with different labels,
     drop every copy, because we can't know which label is right.
  4. Mask first names found during profiling with the token "<name>".
  5. Handle rare classes (classes below MIN_CLASS_SIZE get merged into "Other").
  6. Give every ticket a group_id. Near-duplicate tickets share one, so the
     split in the next stage can keep each cluster on one side.

Short tickets are kept on purpose: "printer broken" is a real ticket.

Run:  python -m src.clean_data
Reads:  data/raw/all_tickets_processed_improved_v3.csv
Writes: data/processed/tickets_clean.csv  (columns: text, label, group_id)
        reports/cleaning_report.md        (rows affected by each rule)
"""
import re

import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, REPORTS_DIR
from src.dedup import assign_groups

RAW_FILE = DATA_RAW / "all_tickets_processed_improved_v3.csv"
CLEAN_FILE = DATA_PROCESSED / "tickets_clean.csv"

# The names the profiling found (see reports/data_profile.md). We mask only
# these, not the whole spot-check list, because some names on that list are
# also ordinary words ("mark", "will", "dan" in other languages).
NAMES_TO_MASK = ["ion", "florin"]
NAME_TOKEN = "<name>"

# No class is below this size today (the smallest has 1,760 rows), so this
# rule changes nothing. It's here so the pipeline stays safe if the data changes.
MIN_CLASS_SIZE = 100
OTHER_LABEL = "Other"


def normalize_whitespace(texts):
    """Strip leading/trailing whitespace and collapse internal runs to one space."""
    return texts.fillna("").str.split().str.join(" ")


def drop_empty(df):
    """Remove rows whose text is an empty string."""
    return df[df["text"] != ""]


def drop_duplicates(df):
    """Keep one copy of each exact-duplicate text; drop texts that have conflicting labels."""
    labels_per_text = df.groupby("text")["label"].transform("nunique")
    df = df[labels_per_text == 1]
    return df.drop_duplicates(subset="text", keep="first")


def mask_names(texts, names=NAMES_TO_MASK, token=NAME_TOKEN):
    """Replace whole-word matches of each name with a placeholder token.

    The \\b word boundaries matter: "ion" must match the name, not the
    "ion" inside "permission" or "version".
    """
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b")
    return texts.str.replace(pattern, token, regex=True)


def merge_rare_classes(labels, min_size=MIN_CLASS_SIZE, other=OTHER_LABEL):
    """Relabel any class with fewer than min_size rows as `other`."""
    counts = labels.value_counts()
    rare = counts[counts < min_size].index
    return labels.where(~labels.isin(rare), other)


def main():
    raw = pd.read_csv(RAW_FILE)
    df = pd.DataFrame({"text": raw["Document"], "label": raw["Topic_group"]})
    log = [("Raw rows", len(df), "")]

    before = df["text"].copy()
    df["text"] = normalize_whitespace(df["text"])
    log.append(("1. Normalise whitespace", len(df), f"{(before.fillna('') != df['text']).sum()} texts changed"))

    n = len(df)
    df = drop_empty(df)
    log.append(("2. Drop empty text", len(df), f"{n - len(df)} rows dropped"))

    n = len(df)
    df = drop_duplicates(df)
    log.append(("3. Drop exact duplicates", len(df), f"{n - len(df)} rows dropped"))

    before = df["text"].copy()
    df["text"] = mask_names(df["text"])
    log.append(("4. Mask first names", len(df), f"{(before != df['text']).sum()} texts changed"))

    before = df["label"].copy()
    df["label"] = merge_rare_classes(df["label"])
    log.append(("5. Merge rare classes", len(df), f"{(before != df['label']).sum()} labels changed"))

    df = df.reset_index(drop=True)
    df["group_id"] = assign_groups(df["text"].tolist())
    sizes = df["group_id"].value_counts()
    in_clusters = sizes[sizes > 1].sum()
    log.append(("6. Assign near-duplicate groups", len(df),
                f"{df['group_id'].nunique():,} groups; {in_clusters:,} rows are in "
                f"{(sizes > 1).sum():,} multi-ticket clusters"))

    df.to_csv(CLEAN_FILE, index=False)

    out = ["# Cleaning report", "", "| Step | Rows after | Effect |", "|---|---:|---|"]
    out += [f"| {step} | {rows:,} | {effect} |" for step, rows, effect in log]
    counts = df["label"].value_counts()
    out += ["", "## Class distribution after cleaning", "| Class | Count | % |", "|---|---:|---:|"]
    out += [f"| {c} | {counts[c]:,} | {counts[c] / len(df) * 100:.2f} |" for c in counts.index]
    out += ["", f"Saved {len(df):,} rows to `{CLEAN_FILE.relative_to(CLEAN_FILE.parents[2])}`"]
    report = "\n".join(out)
    (REPORTS_DIR / "cleaning_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
