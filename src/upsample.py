"""Code to upsample a given dataset to a 1:1 ratio between majority and minority class"""
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CSV_PATH = PROJECT_ROOT / "sentiment_classifier" / "data" / "processed" / "Reviews.csv"

df = pd.read_csv(CSV_PATH)

def upsample(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()

    minority_df = df[df["sentiment"] == "negative"]
    majority_df = df[df["sentiment"] == "positive"]

    minority_upsampled = minority_df.sample(
        n = len(majority_df),
        replace = True,
        random_state = 42
    )

    upsampled_df = pd.concat([minority_upsampled, majority_df])

    return upsampled_df
    
if __name__ == "__main__":
    upsampled_df = upsample(df)
    print("Total length of upsampled dataset:", len(upsampled_df))
    print("Negative Reviews:", (upsampled_df["sentiment"] == "negative").sum())
    print("Positive Reviews:", (upsampled_df["sentiment"] == "positive").sum())
