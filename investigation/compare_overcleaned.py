"""Compare selected normal and overcleaned reviews."""

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NORMAL_PATH = PROJECT_ROOT / "data" / "processed" / "IMDB_Dataset.csv"
OVERCLEANED_PATH = (
    PROJECT_ROOT / "data" / "processed" / "IMDB_Dataset_overcleaned.csv"
)
REVIEW_COLUMN = "review"
SAMPLE_COUNT = 5
SAMPLE_SPACING = 1_000
WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def words_in_review(review: str) -> set[str]:
    """Return normalized words from a review."""
    return set(WORD_PATTERN.findall(review.lower()))


def compare_selected_reviews(
    normal_reviews: pd.DataFrame, overcleaned_reviews: pd.DataFrame
) -> dict[int, set[str]]:
    """Return words removed from five reviews spaced 1,000 rows apart."""
    if len(normal_reviews) != len(overcleaned_reviews):
        raise ValueError("The datasets must contain the same number of reviews")
    if REVIEW_COLUMN not in normal_reviews or REVIEW_COLUMN not in overcleaned_reviews:
        raise ValueError(f"Both datasets must contain a '{REVIEW_COLUMN}' column")

    selected_indices = range(0, SAMPLE_COUNT * SAMPLE_SPACING, SAMPLE_SPACING)
    if max(selected_indices) >= len(normal_reviews):
        raise ValueError("The datasets do not contain enough reviews to sample")

    missing_words_by_index = {}
    for index in selected_indices:
        normal_words = words_in_review(str(normal_reviews.iloc[index][REVIEW_COLUMN]))
        overcleaned_words = words_in_review(
            str(overcleaned_reviews.iloc[index][REVIEW_COLUMN])
        )
        missing_words_by_index[index] = normal_words - overcleaned_words
    return missing_words_by_index


def main() -> None:
    normal_reviews = pd.read_csv(NORMAL_PATH)
    overcleaned_reviews = pd.read_csv(OVERCLEANED_PATH)

    missing_words_by_index = compare_selected_reviews(
        normal_reviews, overcleaned_reviews
    )
    for index, missing_words in missing_words_by_index.items():
        print(f"Review {index}: {sorted(missing_words)}")


if __name__ == "__main__":
    main()
