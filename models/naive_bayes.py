"""Create Multimonial Naive Bayes Pipeline with tuned alpha, prints f1-scores and accuracy"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# pathfinding to the csv file
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CSV_PATH = PROJECT_ROOT / "sentiment_classifier" / "data" / "processed" / "trimmed_IMDB_Dataset.csv"

trimmed_reviews = pd.read_csv(CSV_PATH)


def naive_bayes(trimmed_reviews = trimmed_reviews):
    """Create the Multimonial Naive Bayes Pipeline using trimmed_IMDB_Dataset and prints f1 scores and accuracy"""

    # Initialize feature (X), or the input(s) of the model; labels (y), or the output(s) of the model
    X = trimmed_reviews["review"]
    y = trimmed_reviews["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 42,
        stratify = y
    )

    model_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('model', MultinomialNB(alpha = 0.5))
    ])

    model_pipeline.fit(X_train, y_train) # train model on vectorized training data, calculates probabilities for each sentiment class
    y_prediction = model_pipeline.predict(X_test) # after training, model generates predictions for the test set

    results_df = pd.DataFrame({
        'Review': X_test,
        'Actual Label': y_test,
        'Predicted Label': y_prediction
    })

    # Checking word associations using P(class | word) for every word in the vocabulary list of training data

    vectorizer = model_pipeline.named_steps["vectorizer"]
    model = model_pipeline.named_steps["model"]
    vocab_list = vectorizer.get_feature_names_out()
    print("Vocab_list:", vocab_list)
    prob_word_given_class = np.exp(model.feature_log_prob_)
    class_priors = np.exp(model.class_log_prior_)
    p_neg = class_priors[0]
    p_pos = class_priors[1]

    p_word_given_neg = prob_word_given_class[0]
    p_word_given_pos = prob_word_given_class[1]

    # Use law of total probabilty to calculate P(word) for Bayes formula
    p_word = (p_word_given_pos * p_pos) + (p_word_given_neg * p_neg)

    # Calculate P(Positive | word) and P(Negative | word)
    p_neg_given_word = (p_word_given_neg * p_neg) / p_word
    p_pos_given_word = (p_word_given_pos * p_pos) / p_word
    sorted_p_neg_given_word = np.argsort(-p_neg_given_word)
    sorted_p_pos_given_word = np.argsort(-p_pos_given_word)
    
    print(p_neg_given_word, p_pos_given_word)

    for neg_idx in sorted_p_neg_given_word[0:20]:
        print("Top negative words:", vocab_list[neg_idx], f"P(Negative | {vocab_list[neg_idx]}) = {p_neg_given_word[neg_idx]}")
    print("\n")
    for pos_idx in sorted_p_pos_given_word[0:20]:
        print("Top positive words:", vocab_list[pos_idx], f"P(Positive | {vocab_list[pos_idx]}) = {p_pos_given_word[pos_idx]}")

    print("Classes at index positions in array:", model.classes_)

    # Quick Check: Display the first 10 rows
    print("\nSample")
    print(results_df.head(20))

    print(classification_report(y_test, y_prediction))


if __name__ == "__main__":
    naive_bayes(trimmed_reviews)
    