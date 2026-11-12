"""
results from running it:
                  reviews: 363850
           positive share: 0.843
 always-positive accuracy: 0.843
                 accuracy: 0.876
                precision: 0.898
                   recall: 0.962
            F1 (positive): 0.929
            F1 (negative): 0.508
                 macro F1: 0.718
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_PATH = "data/raw/Amazon_Reviews.csv"


def load_amazon(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["UserId", "Text"]) #Same people were making the smae reviews so clean duplicates
    df = df[df["Score"] != 3].copy()
    df["label"] = (df["Score"] >= 4).astype(int)
    return df


def evaluate_vader(df, text_column, label_column, threshold=0.0):
    analyzer = SentimentIntensityAnalyzer()
    compound = df[text_column].apply(lambda text: analyzer.polarity_scores(text)["compound"])
    y_pred = (compound >= threshold).astype(int).to_numpy()
    y_true = df[label_column].to_numpy()

    return {
        "reviews": len(y_true),
        "positive share": y_true.mean(),
        "always-positive accuracy": y_true.mean(),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "F1 (positive)": f1_score(y_true, y_pred),
        "F1 (negative)": f1_score(y_true, y_pred, pos_label=0),
        "macro F1": f1_score(y_true, y_pred, average="macro"),
    }


def main():
    df = load_amazon()
    print(f"VADER is being run on {len(df)} reviews :)")
    results = evaluate_vader(df, "Text", "label")
    for name, value in results.items():
        print(f"{name:>25}: {value:.3f}" if isinstance(value, float) else f"{name:>25}: {value}")


if __name__ == "__main__":
    main()