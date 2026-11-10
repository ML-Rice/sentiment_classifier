"""Estimate word-level KernelSHAP effects for the fine-tuned DistilBERT model."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.distilbert import MAX_LENGTH, load_and_split_data, load_config  # noqa: E402


DEFAULT_BATCH_SIZE = 150
MAX_REVIEWS = 1_000


class CachedProbabilityPredictor:
    """Batch model calls and retain probabilities for masked texts in one review."""

    def __init__(self, model, tokenizer, device, batch_size=DEFAULT_BATCH_SIZE):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.cache = {}

    def clear_cache(self):
        """Discard predictions after completing a review's SHAP calculation."""
        self.cache.clear()

    def __call__(self, texts):
        # KernelSHAP can supply repeated coalitions within a review.
        # Deduplicate before tokenization so each unique text receives one model prediction.
        texts = [str(text) for text in texts]
        missing = []
        seen = set()
        for text in texts:
            if text not in self.cache and text not in seen:
                missing.append(text)
                seen.add(text)

        if missing:
            self._predict_missing(missing)

        return np.asarray([self.cache[text] for text in texts])

    def _predict_missing(self, texts):
        self.model.eval()
        for start in range(0, len(texts), self.batch_size):
            encoded = self.tokenizer(
                texts[start : start + self.batch_size],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=(self.device == "cuda")
            ):
                logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            self.cache.update(zip(texts[start : start + self.batch_size], probabilities))


def predict_from_masks(masks, words, tokenizer, predictor):
    texts = []
    for mask in masks:
        kept = [word if keep else tokenizer.mask_token for word, keep in zip(words, mask)]
        texts.append(" ".join(kept))
    return predictor(texts)


def explain_review(text, tokenizer, predictor):
    words = text.split()
    background = np.zeros((1, len(words)))  # all 0s: every word hidden
    explainer = shap.KernelExplainer(
        lambda masks: predict_from_masks(masks, words, tokenizer, predictor),
        background,
    )
    shap_values = explainer.shap_values(
        np.ones((1, len(words))), nsamples="auto", l1_reg=False, silent=True
    )
    base_value = float(np.squeeze(explainer.expected_value))
    return pd.DataFrame({"word": words, "shap": shap_values[0]}), base_value


def explain_reviews(reviews_df, tokenizer, predictor, seed=42):
    np.random.seed(seed)  # KernelSHAP samples word combinations randomly; this makes runs repeatable
    all_word_rows = []

    for review_number, row in enumerate(reviews_df.itertuples(), start=1):
        # Masked texts are unique to a review, so retaining prior reviews' predictions
        # only grows memory without avoiding any model calls.
        predictor.clear_cache()
        try:
            word_table, _ = explain_review(row.review, tokenizer, predictor)
            cached_predictions = len(predictor.cache)
        finally:
            # Release this review's cache even if SHAP raises partway through it.
            predictor.clear_cache()

        word_table["review_id"] = row.Index
        word_table["label"] = row.label
        all_word_rows.append(word_table)
        print(
            f"explained review {review_number}/{len(reviews_df)}  "
            f"cached predictions for review: {cached_predictions}"
        )

    return pd.concat(all_word_rows, ignore_index=True)


def systematically_sample_reviews(reviews_df, max_reviews=MAX_REVIEWS):
    """Select evenly spaced reviews, retaining every review for smaller datasets."""
    if len(reviews_df) <= max_reviews:
        return reviews_df

    positions = np.linspace(0, len(reviews_df) - 1, num=max_reviews, dtype=int)
    return reviews_df.iloc[positions]


def summarize_words(word_impacts, min_reviews=3):
    impacts = word_impacts.copy()
    impacts["clean_word"] = impacts["word"].str.lower().str.replace(r"[^a-z']", "", regex=True)
    impacts = impacts[impacts["clean_word"] != ""]

    summary = (
        impacts.groupby("clean_word")
        .agg(
            mean_shap=("shap", "mean"),
            std_shap=("shap", "std"),
            mean_abs_shap=("shap", lambda values: values.abs().mean()),
            occurrences=("shap", "size"),
            reviews=("review_id", "nunique"),
        )
        .reset_index()
    )

    frequent = summary[summary["reviews"] >= min_reviews]
    return frequent.sort_values("mean_shap", ascending=False).reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--min-reviews", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    seed = int(config["random_seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = PROJECT_ROOT / "models" / "distilbert_finetuned"
    output_dir = PROJECT_ROOT / "data" / "processed"

    print(f"Using device: {device}")
    print(f"Loading fine-tuned model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Compiled model with torch.compile")
    except RuntimeError as error:
        # Compilation is an optimization; keep the long SHAP job runnable on unsupported setups.
        print(f"torch.compile unavailable; using eager model ({error})")
    predictor = CachedProbabilityPredictor(model, tokenizer, device, args.batch_size)

    train_df, validation_df, test_df = load_and_split_data(config)
    # The splits are only used by fine-tuning; sample evenly across the full dataset
    # for attribution rather than overrepresenting any contiguous portion of it.
    all_reviews = pd.concat([train_df, validation_df, test_df]).sort_index()
    reviews = systematically_sample_reviews(all_reviews)
    print(f"Explaining {len(reviews)} systematically sampled reviews from {len(all_reviews)} total")
    word_impacts = explain_reviews(reviews, tokenizer, predictor, seed)
    summary = summarize_words(word_impacts, args.min_reviews)

    # Restrict both rankings to words observed often enough for a meaningful review-level comparison.
    columns = ["clean_word", "mean_shap", "std_shap"]
    top_words = summary.head(20)[columns].rename(columns={"clean_word": "word"})
    bottom_words = summary.tail(20).sort_values("mean_shap")[columns].rename(
        columns={"clean_word": "word"}
    )
    top_path = output_dir / "distilbert_kernelshap_top_20_words.csv"
    bottom_path = output_dir / "distilbert_kernelshap_bottom_20_words.csv"
    top_words.to_csv(top_path, index=False)
    bottom_words.to_csv(bottom_path, index=False)

    print("Top 20 words by mean SHAP value:")
    print(top_words.to_string(index=False))
    print("\nBottom 20 words by mean SHAP value:")
    print(bottom_words.to_string(index=False))
    print(f"\nWrote {top_path}")
    print(f"Wrote {bottom_path}")


if __name__ == "__main__":
    main()
