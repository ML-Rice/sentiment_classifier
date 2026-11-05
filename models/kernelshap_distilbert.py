import numpy as np
import pandas as pd
import shap
import torch


def predict_probabilities(model, tokenizer, texts, device, batch_size=32, max_length=512):
    model.eval()
    probabilities = []

    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            scores = model(**encoded).logits

        probabilities.append(torch.softmax(scores.float(), dim=1)[:, 1].cpu().numpy())

    return np.concatenate(probabilities)


def predict_from_masks(masks, words, model, tokenizer, device):
    texts = []
    for mask in masks:
        kept = [word if keep else tokenizer.mask_token for word, keep in zip(words, mask)]
        texts.append(" ".join(kept))
    return predict_probabilities(model, tokenizer, texts, device)


def explain_review(text, model, tokenizer, device):
    words = text.split()
    background = np.zeros((1, len(words)))  # all 0s: every word hidden
    explainer = shap.KernelExplainer(
        lambda masks: predict_from_masks(masks, words, model, tokenizer, device),
        background,
    )
    shap_values = explainer.shap_values(
        np.ones((1, len(words))), nsamples="auto", l1_reg=False, silent=True
    )
    base_value = float(np.squeeze(explainer.expected_value))
    return pd.DataFrame({"word": words, "shap": shap_values[0]}), base_value


def choose_sample(test_df, max_words=150, n_reviews=40, seed=42):
    word_counts = test_df["review"].str.split().str.len()
    return test_df[word_counts <= max_words].sample(n=n_reviews, random_state=seed)


def explain_reviews(reviews_df, model, tokenizer, device, prediction_column="distilbert_pred", seed=42):
    np.random.seed(seed)  # KernelSHAP samples word combinations randomly; this makes runs repeatable
    all_word_rows = []

    for review_number, (row_index, row) in enumerate(reviews_df.iterrows(), start=1):
        word_table, _ = explain_review(row["review"], model, tokenizer, device)
        word_table["review_id"] = row_index
        word_table["label"] = row["label"]
        word_table["prediction"] = row[prediction_column]
        all_word_rows.append(word_table)
        print(f"explained review {review_number}/{len(reviews_df)}")

    return pd.concat(all_word_rows, ignore_index=True)


def summarize_words(word_impacts, min_reviews=3):
    impacts = word_impacts.copy()
    impacts["clean_word"] = impacts["word"].str.lower().str.replace(r"[^a-z']", "", regex=True)
    impacts = impacts[impacts["clean_word"] != ""]

    summary = (
        impacts.groupby("clean_word")
        .agg(
            mean_shap=("shap", "mean"),
            mean_abs_shap=("shap", lambda values: values.abs().mean()),
            occurrences=("shap", "size"),
            reviews=("review_id", "nunique"),
        )
        .reset_index()
    )

    frequent = summary[summary["reviews"] >= min_reviews]
    return frequent.sort_values("mean_abs_shap", ascending=False).round(4).reset_index(drop=True)