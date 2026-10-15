"""Remove words with neutral word tilts from the cleaned dataset."""

import re
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from investigation import word_tilt


def overclean_reviews(
    dataframe: pd.DataFrame, words_to_remove: set[str]
) -> pd.DataFrame:
    """Remove supplied words from every review using vectorized string operations."""
    overcleaned = dataframe.copy()
    escaped_words = sorted(
        (re.escape(word) for word in words_to_remove), key=len, reverse=True
    )
    if not escaped_words:
        return overcleaned

    words_pattern = rf"\b(?:{'|'.join(escaped_words)})\b"
    overcleaned["review"] = (
        overcleaned["review"]
        .fillna("")
        .astype(str)
        .str.replace(words_pattern, "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return overcleaned


def main() -> None:
    config_path = Path(__file__).resolve().with_name("config.yaml")
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    dataset_path = Path(config["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    processed_path = PROJECT_ROOT / "data" / "processed" / dataset_path.name
    output_path = processed_path.with_name(
        f"{processed_path.stem}_overcleaned{processed_path.suffix}"
    )

    reviews = word_tilt.load_reviews(dataset_path)
    word_tilts = word_tilt.calculate_word_tilts(reviews)
    words_to_remove = word_tilt.words_with_tilt_near_one(word_tilts)

    cleaned_dataset = pd.read_csv(processed_path)
    overcleaned_dataset = overclean_reviews(cleaned_dataset, words_to_remove)
    overcleaned_dataset.to_csv(output_path, index=False)
    print(f"Saved overcleaned dataset to {output_path}")


if __name__ == "__main__":
    main()
