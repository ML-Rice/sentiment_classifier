from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "Reviews.csv"
REPORTS_DIR = ROOT / "reports"
OUT_PATH = ROOT / "data" / "processed" / "Amazon_Reviews_imputed.csv"

KNN_SAMPLE_SIZE = 50_000
SEED = 42


def check_missingness(df):
    print("=" * 60)
    print("MISSING VALUES PER COLUMN")
    print("=" * 60)
    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(3),
    })
    print(report)
    flagged = report[report["missing_pct"] > 30]
    print("\nColumns over 30% missing (lead's rule):",
          list(flagged.index) if len(flagged) else "none")
    print("\nBlank (empty or whitespace-only) text:")
    for col in ["Summary", "Text"]:
        blank = df[col].fillna("").str.strip().eq("").sum()
        print(f"  {col}: {blank}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORTS_DIR / "missingness_report.csv")
    return report


def build_helpful_ratio(df):
    print("\n" + "=" * 60)
    print("STEP 2: HELPFULNESS RATIO")
    print("=" * 60)
    num = df["HelpfulnessNumerator"]
    den = df["HelpfulnessDenominator"]

    no_votes = den == 0
    impossible = num > den
    valid = (den > 0) & (num <= den)

    print(f"Rows with 0 votes (ratio undef): {no_votes.sum()} "
          f"({no_votes.mean() * 100:.1f}%)")
    print(f"Rows with numerator > denominator (impossible): {impossible.sum()}")

    df["has_votes"] = (den > 0).astype(int)
    df["helpful_ratio"] = np.where(valid, num / den.where(den > 0), np.nan)

    pct = df["helpful_ratio"].isna().mean() * 100
    print(f"helpful_ratio missing after build: {pct:.1f}%")
    if pct > 30:
        print(" Over the 30% rule, but there is an obvious reason: "
              "nobody voted, so the ratio does not exist.")
    print("\nDoes 'no votes' relate to score? (mean Score by group)")
    print(df.groupby("has_votes")["Score"].agg(["mean", "count"]).round(3))
    return df


def mean_impute(df):
    mean_val = df["helpful_ratio"].mean()
    df["helpful_ratio_mean"] = df["helpful_ratio"].fillna(mean_val)
    print("\n" + "=" * 60)
    print("STEP 3a: MEAN IMPUTATION")
    print("=" * 60)
    print(f"Mean of observed helpful_ratio: {mean_val:.4f}")
    return df


def knn_impute_sample(df):
    print("\n" + "=" * 60)
    print(f"STEP 3b: KNN IMPUTATION (sample of {KNN_SAMPLE_SIZE})")
    print("=" * 60)
    cols = ["Score", "Time", "HelpfulnessDenominator", "helpful_ratio"]
    n = min(KNN_SAMPLE_SIZE, len(df))
    sample = df.sample(n, random_state=SEED)[cols + ["helpful_ratio_mean"]].copy()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(sample[cols])
    imputed = KNNImputer(n_neighbors=5).fit_transform(scaled)
    imputed = scaler.inverse_transform(imputed)

    sample["helpful_ratio_knn"] = imputed[:, cols.index("helpful_ratio")]

    was_missing = sample["helpful_ratio"].isna()
    print(f"Imputed {was_missing.sum()} missing values in the sample")
    print("\nImputed values only (mean strategy vs KNN strategy):")
    compare = sample.loc[was_missing, ["helpful_ratio_mean", "helpful_ratio_knn"]]
    print(compare.describe().round(4))

    print("\nObserved (real) helpful_ratio for reference:")
    print(sample["helpful_ratio"].describe().round(4))

    sample.to_csv(REPORTS_DIR / "knn_vs_mean_sample.csv", index=False)


def main():
    df = pd.read_csv(RAW_PATH)
    print("Columns:", df.columns.tolist())
    print("Rows:", len(df), "\n")

    check_missingness(df)
    df = build_helpful_ratio(df)
    df = mean_impute(df)
    knn_impute_sample(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
