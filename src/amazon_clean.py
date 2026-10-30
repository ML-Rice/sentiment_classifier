from pathlib import Path

import pandas as pd

from clean import clean_text

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "Reviews.csv"
OUT_PATH = ROOT / "data" / "processed" / "Amazon_Reviews_clean.csv"


def main():
    df = pd.read_csv(RAW_PATH)
    print("Columns:", df.columns.tolist())
    print("Starting rows:", len(df))

    print("\nMissing data (%):")
    print((df.isna().mean() * 100).round(2))

    df = df.drop(columns=["Id", "ProfileName"])
    before = len(df)
    df = df.drop_duplicates(subset=["UserId", "Time", "Text"])
    print(f"\nDuplicates removed: {before - len(df)}")

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
