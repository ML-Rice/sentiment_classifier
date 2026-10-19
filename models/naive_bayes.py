"""Create Multimonial Naive Bayes Pipeline with tuned alpha, prints f1-scores and accuracy"""

from pathlib import Path
import pandas as pd
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

    # Quick Check: Display the first 10 rows
    print("Sample")
    print(results_df.head(20))

    print(classification_report(y_test, y_prediction))


if __name__ == "__main__":
    naive_bayes(trimmed_reviews)