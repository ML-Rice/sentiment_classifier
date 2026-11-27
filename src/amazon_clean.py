import html
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords

from dedupe import remove_duplicates

WORDS_TO_KEEP = [
    "no", "nor", "not", "never", "none", "nobody", "nothing", "nowhere",
    "neither", "cannot", "without",
    "but", "however", "although", "though", "yet", "except", "despite",
    "against",
    "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn",
    "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won",
    "wouldn", "don", "ain",
]

SPECIAL_CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "shan't": "shall not",
    "ain't": "is not",
    "let's": "let us",
    "i'm": "i am",
}

CONTRACTION_ENDINGS = [
    ("n't", " not"),
    ("'re", " are"),
    ("'ve", " have"),
    ("'ll", " will"),
    ("'d", " would"),
    ("'m", " am"),
    ("'s", " is"),
]


def get_stop_words():
    try:
        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("english")
    return {w for w in words if w not in WORDS_TO_KEEP}


STOP_WORDS = get_stop_words()


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)

    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    for short, long in SPECIAL_CONTRACTIONS.items():
        text = text.replace(short, long)
    for ending, replacement in CONTRACTION_ENDINGS:
        text = text.replace(ending, replacement)
    text = re.sub(r"[^a-z ]", " ", text)
    return " ".join(w for w in text.split() if w not in STOP_WORDS)


ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "Reviews.csv"
OUT_PATH = ROOT / "data" / "processed" / "Amazon_Reviews_clean.csv"


def main():
    df = pd.read_csv(RAW_PATH)
    print("Columns:", df.columns.tolist())
    print("Starting rows:", len(df))

    print("\nMissing data (%):")
    print((df.isna().mean() * 100).round(2))

    print()
    df = remove_duplicates(df)
    df = df.drop(columns=["Id", "ProfileName"])

    before = len(df)
    df = df[df["HelpfulnessNumerator"] <= df["HelpfulnessDenominator"]]
    print(f"Impossible helpfulness rows removed: {before - len(df)}")

    before = len(df)
    df = df.dropna(subset=["Text", "Summary"]).copy()
    print(f"Rows with missing text/summary removed: {before - len(df)}")

    df["Time"] = pd.to_datetime(df["Time"], unit="s")

    df["label"] = (df["Score"] >= 3).astype(int)
    print("\nClass balance (1 = positive):")
    print(df["label"].value_counts(normalize=True).round(3))

    df["has_votes"] = (df["HelpfulnessDenominator"] > 0).astype(int)
    df["helpful_ratio"] = df["HelpfulnessNumerator"] / df["HelpfulnessDenominator"]
    print(f"\nhelpful_ratio missing (no votes): "
          f"{df['helpful_ratio'].isna().mean() * 100:.1f}%")
    df["helpful_ratio_mean"] = df["helpful_ratio"].fillna(df["helpful_ratio"].mean())

    print("\nCleaning text (this takes a few minutes)...")
    df["text_clean"] = df["Text"].map(clean_text)
    df["summary_clean"] = df["Summary"].map(clean_text)
    before = len(df)
    df = df[(df["text_clean"] != "") & (df["summary_clean"] != "")]
    print(f"Rows empty after cleaning removed: {before - len(df)}")

    print("Final rows:", len(df))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
