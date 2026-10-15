from pathlib import Path

import pandas as pd
import yaml
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


N = 50000
ANALYZER = SentimentIntensityAnalyzer()


def predict(text: str) -> int:
    scores = ANALYZER.polarity_scores(text)
    return int(scores["compound"] >= 0)


def calculate_metrics(y_true, y_pred) -> dict[str, float | int]:
    """Calculate classification metrics and confusion-matrix counts."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    accuracy = (tn + tp) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (recall + precision)
        if recall + precision
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_dataset(dataset_path: Path) -> dict[str, float | int]:
    """Run VADER predictions and calculate metrics for one processed dataset."""
    dataframe = pd.read_csv(dataset_path)
    y_true = (dataframe["sentiment"] == "positive").astype(int).head(N).to_numpy()
    y_pred = dataframe["review"].head(N).apply(predict).to_numpy()
    return calculate_metrics(y_true, y_pred)


def print_metrics(dataset_name: str, metrics: dict[str, float | int]) -> None:
    """Print the VADER metrics for one dataset."""
    print(f"{dataset_name}:")
    print(
        f"Accuracy: {metrics['accuracy']:.3f}, "
        f"Precision: {metrics['precision']:.3f}, "
        f"Recall: {metrics['recall']:.3f}, "
        f"F1 score: {metrics['f1']:.3f}"
    )
    print(
        f"TP: {metrics['tp']}, TN: {metrics['tn']}, "
        f"FP: {metrics['fp']}, FN: {metrics['fn']}"
    )


def main() -> None:
    config_path = Path(__file__).resolve().with_name("config.yaml")
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    raw_dataset_path = Path(config["dataset_path"])
    processed_directory = config_path.parent.parent / "data" / "processed"
    processed_dataset_path = processed_directory / f"{raw_dataset_path.stem}.csv"
    overcleaned_dataset_path = processed_directory / (
        f"{raw_dataset_path.stem}_overcleaned.csv"
    )

    print_metrics(
        processed_dataset_path.name,
        evaluate_dataset(processed_dataset_path),
    )
    print_metrics(
        overcleaned_dataset_path.name,
        evaluate_dataset(overcleaned_dataset_path),
    )


if __name__ == "__main__":
    main()
