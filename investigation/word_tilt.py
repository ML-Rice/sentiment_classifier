"""Calculate how strongly words are associated with positive reviews."""

from collections import Counter
from pathlib import Path
import re

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "IMDB_Dataset.csv"
)
MIN_REVIEWS = 5
WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def load_reviews(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the IMDb reviews CSV."""
    reviews = pd.read_csv(path)
    required_columns = {"review", "sentiment"}
    missing_columns = required_columns - set(reviews.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required column(s): {missing}")

    reviews = reviews.loc[:, ["review", "sentiment"]].copy()
    reviews["review"] = reviews["review"].fillna("").astype(str)
    reviews["sentiment"] = reviews["sentiment"].str.lower()
    unexpected_sentiments = set(reviews["sentiment"]) - {"positive", "negative"}
    if unexpected_sentiments:
        unexpected = ", ".join(sorted(unexpected_sentiments))
        raise ValueError(f"Unexpected sentiment value(s): {unexpected}")
    return reviews


def _words_in_review(review: str) -> set[str]:
    """Return unique lowercase words from one review."""
    review_without_html = HTML_TAG_PATTERN.sub(" ", review.lower())
    return set(WORD_PATTERN.findall(review_without_html))


def calculate_word_tilts(
    reviews: pd.DataFrame, min_reviews: int = MIN_REVIEWS
) -> pd.DataFrame:
    """Calculate positive-to-negative probability ratios for sufficiently common words."""
    required_columns = {"review", "sentiment"}
    missing_columns = required_columns - set(reviews.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Reviews are missing required column(s): {missing}")
    if min_reviews < 1:
        raise ValueError("min_reviews must be at least 1")

    positive_reviews = reviews[reviews["sentiment"] == "positive"]
    negative_reviews = reviews[reviews["sentiment"] == "negative"]
    if positive_reviews.empty or negative_reviews.empty:
        raise ValueError("The dataset must contain both positive and negative reviews")

    positive_word_counts = Counter(
        word
        for review in positive_reviews["review"]
        for word in _words_in_review(str(review))
    )
    negative_word_counts = Counter(
        word
        for review in negative_reviews["review"]
        for word in _words_in_review(str(review))
    )

    eligible_words = sorted(
        word
        for word in positive_word_counts.keys() & negative_word_counts.keys()
        if positive_word_counts[word] >= min_reviews
        and negative_word_counts[word] >= min_reviews
    )
    positive_review_total = len(positive_reviews)
    negative_review_total = len(negative_reviews)

    result = pd.DataFrame(
        {
            "word": eligible_words,
            "positive_review_count": [
                positive_word_counts[word] for word in eligible_words
            ],
            "negative_review_count": [
                negative_word_counts[word] for word in eligible_words
            ],
        }
    )
    result["positive_probability"] = (
        result["positive_review_count"] / positive_review_total
    )
    result["negative_probability"] = (
        result["negative_review_count"] / negative_review_total
    )
    result["word_tilt"] = (
        result["positive_probability"] / result["negative_probability"]
    )
    return result.sort_values("word_tilt", ascending=False, ignore_index=True)


def main() -> None:
    reviews = load_reviews()
    word_tilts = calculate_word_tilts(reviews).head(20)
    print(word_tilts.to_string(index=False))


if __name__ == "__main__":
    main()