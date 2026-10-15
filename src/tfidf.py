"""Create TF-IDF features for the cleaned and overcleaned datasets."""

from pathlib import Path
import pickle

import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def load_config() -> dict:
    """Load the project configuration."""
    config_path = Path(__file__).resolve().with_name("config.yaml")
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping")
    return config


def load_dataset_paths(config: dict) -> tuple[Path, Path]:
    """Return the normal and overcleaned processed dataset paths."""
    try:
        configured_path = Path(config["dataset_path"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            "config.yaml must define a dataset_path value"
        ) from error

    dataset_name = configured_path.name
    processed_path = PROCESSED_DIRECTORY / dataset_name
    overcleaned_path = processed_path.with_name(
        f"{processed_path.stem}_overcleaned{processed_path.suffix}"
    )
    return processed_path, overcleaned_path


def fit_and_save_tfidf(
    dataset_path: Path,
    output_stem: str,
    test_size: float,
    random_seed: int,
) -> None:
    """Split one dataset, fit TF-IDF, and save sparse features and labels."""
    dataframe = pd.read_csv(dataset_path)
    required_columns = {"review", "sentiment"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{dataset_path} is missing required column(s): {missing}")

    train, test = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_seed,
        stratify=dataframe["sentiment"],
    )

    train_reviews = train["review"].fillna("").astype(str)
    test_reviews = test["review"].fillna("").astype(str)
    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train_reviews)
    test_features = vectorizer.transform(test_reviews)

    train_features_path = PROCESSED_DIRECTORY / f"{output_stem}_train_X.npz"
    test_features_path = PROCESSED_DIRECTORY / f"{output_stem}_test_X.npz"
    train_labels_path = PROCESSED_DIRECTORY / f"{output_stem}_train_y.csv"
    test_labels_path = PROCESSED_DIRECTORY / f"{output_stem}_test_y.csv"
    preprocessor_path = PROCESSED_DIRECTORY / f"{output_stem}_preprocessor.pkl"
    save_npz(train_features_path, train_features)
    save_npz(test_features_path, test_features)
    train[["sentiment"]].to_csv(train_labels_path, index=False)
    test[["sentiment"]].to_csv(test_labels_path, index=False)
    with preprocessor_path.open("wb") as preprocessor_file:
        pickle.dump(vectorizer, preprocessor_file)

    print(f"Saved TF-IDF training features to {train_features_path}")
    print(f"Saved TF-IDF test features to {test_features_path}")
    print(f"Saved training labels to {train_labels_path}")
    print(f"Saved test labels to {test_labels_path}")
    print(f"Saved fitted preprocessor to {preprocessor_path}")


def main() -> None:
    """Create TF-IDF artifacts for both processed dataset variants."""
    config = load_config()
    try:
        random_seed = int(config["random_seed"])
        test_size = float(config["test_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "config.yaml must define numeric random_seed and test_size values"
        ) from error
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    dataset_path, overcleaned_path = load_dataset_paths(config)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")
    if not overcleaned_path.is_file():
        raise FileNotFoundError(
            f"Overcleaned dataset not found: {overcleaned_path}"
        )

    fit_and_save_tfidf(
        dataset_path,
        f"{dataset_path.stem}_tfidf",
        test_size,
        random_seed,
    )
    fit_and_save_tfidf(
        overcleaned_path,
        f"{overcleaned_path.stem.removesuffix('_overcleaned')}_tfidf_overcleaned",
        test_size,
        random_seed,
    )


if __name__ == "__main__":
    main()