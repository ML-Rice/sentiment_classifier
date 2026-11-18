from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "processed" / "Reviews.csv"
REPORTS = ROOT / "reports"
TOP_N = 20
MIN_COUNT = 500  # only rank words seen at least this many times in training, so rare words don't dominate


def train_sunny_model(reviews):
    """Reproduce Sunny's training: same split, pipeline and alpha search. Return the best model."""
    X_train, X_test, y_train, y_test = train_test_split(
        reviews["Text"], reviews["sentiment"], test_size=0.2, random_state=42, stratify=reviews["sentiment"]
    )
    pipeline = Pipeline([("vectorizer", CountVectorizer()), ("model", MultinomialNB(alpha=1))])
    search = GridSearchCV(
        pipeline,
        {"model__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Best alpha: {search.best_params_['model__alpha']}")
    print(classification_report(y_test, search.best_estimator_.predict(X_test)))
    return search.best_estimator_


def word_log_ratios(model):
    """For every word: log P(word | positive) - log P(word | negative), plus its training counts."""
    vectorizer = model.named_steps["vectorizer"]
    naive_bayes = model.named_steps["model"]
    classes = list(naive_bayes.classes_)
    positive, negative = classes.index("positive"), classes.index("negative")

    table = pd.DataFrame({
        "word": vectorizer.get_feature_names_out(),
        "log_ratio": naive_bayes.feature_log_prob_[positive] - naive_bayes.feature_log_prob_[negative],
        "count_positive": naive_bayes.feature_count_[positive].astype(int),
        "count_negative": naive_bayes.feature_count_[negative].astype(int),
    })
    table["count_total"] = table["count_positive"] + table["count_negative"]
    return table


def main():
    reviews = pd.read_csv(CSV_PATH)
    model = train_sunny_model(reviews)
    table = word_log_ratios(model)

    frequent = table[table["count_total"] >= MIN_COUNT].sort_values("log_ratio", ascending=False)
    top = frequent.head(TOP_N)
    bottom = frequent.tail(TOP_N).iloc[::-1]

    print(f"{len(table)} words in vocabulary; {len(frequent)} appear at least {MIN_COUNT} times")
    print(f"\nTop {TOP_N} (most typical of POSITIVE reviews):")
    print(top.to_string(index=False))
    print(f"\nBottom {TOP_N} (most typical of NEGATIVE reviews):")
    print(bottom.to_string(index=False))

    output = pd.concat([top.assign(direction="positive"), bottom.assign(direction="negative")])
    output_path = REPORTS / "naive_bayes_Reviews_top_bottom_words.csv"
    output.to_csv(output_path, index=False)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()