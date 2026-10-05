# Sentiment Classifier

Classify movie or product reviews as positive or negative using traditional
natural language processing techniques. The project compares a rule-based
sentiment baseline with scikit-learn pipelines:

- VADER or TextBlob sentiment scoring
- Bag-of-Words with Multinomial Naive Bayes
- TF-IDF with Logistic Regression
- DistilBERT (stretch)

Neutral sentiment can be added later (stretch)

## Project structure

```text
data/
├── raw/                 # Original, unmodified datasets
└── processed/           # Cleaned data and fixed train/test splits
src/
├── config.yaml          # configurations
├── data_loader.py       # Load data and create a reproducible split
├── preprocess.py        # Lowercase, clean, and handle negation-aware stop words
├── baseline.py          # Run VADER/TextBlob baseline predictions
├── train.py             # Train the chosen model
├── evaluate.py          # Calculate metrics and compare pipelines
└── predict.py           # Classify new review text
models/                  # Saved trained pipelines
reports/                 # Metrics, plots, and error analysis
```

The `data/`, `models/`, and `reports/` directories contain `.gitkeep` files so
the empty project structure is preserved in Git.

## Recommended datasets

- IMDb Large Movie Review Dataset
- Amazon product reviews

Place downloaded source files in `data/raw/`. Keep raw data unchanged and
write cleaned or split data to `data/processed/`.

## Setup

Create and activate a virtual environment, then install the project
dependencies:

```bash
uv sync
```

## Workflow

The training script should save fitted pipelines to `models/`. Evaluation
outputs should be written to `reports/`, including:

- Accuracy, precision, recall, and F1 score
- Confusion matrices
- Misclassified examples
- Important features for each trained model

## Reproducibility

Use the fixed train/test split implemented by `src/data_loader.py` and keep
preprocessing inside the same pipeline as model training. This prevents test
data from influencing training and makes comparisons between approaches
consistent.
