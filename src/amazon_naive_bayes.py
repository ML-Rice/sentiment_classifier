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
CSV_PATH = PROJECT_ROOT / "sentiment_classifier" / "data" / "processed" / "Reviews.csv"

trimmed_reviews = pd.read_csv(CSV_PATH)


def naive_bayes(trimmed_reviews = trimmed_reviews):
    # Initialize feature (X), or the input(s) of the model; labels (y), or the output(s) of the model
    X = trimmed_reviews["Text"]
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
        ('model', MultinomialNB(alpha = 1))
    ])

    param_grid = {
        'model__alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    }

    # 3. Initialize GridSearchCV (5-fold cross-validation)
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,                  # 5-fold cross validation
        scoring='f1_macro',    # Metric used to rank parameters
        n_jobs=-1              # Use all CPU cores for faster search
    )

    # 4. Run grid search ONLY on the training data
    grid_search.fit(X_train, y_train)

    # PRINT EACH ALPHA'S CV SCORE HERE:
    print("--- Grid Search Results for Each Alpha ---")
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    params = grid_search.cv_results_['params']

    for mean, std, param in zip(means, stds, params):
        alpha_val = param['model__alpha']
        print(f"Alpha: {alpha_val:<5} | Mean CV F1-Score: {mean:.4f} (+/- {std * 2:.4f})")
    print("------------------------------------------\n")

    print(f"Best Alpha Found: {grid_search.best_params_['model__alpha']}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}\n")

    # 5. Predict on the test set using the best estimator
    best_model = grid_search.best_estimator_
    y_prediction = best_model.predict(X_test)

    # Print test results
    print("--- Classification Report (Tuned Model) ---")
    print(classification_report(y_test, y_prediction))


if __name__ == "__main__":
    naive_bayes(trimmed_reviews)