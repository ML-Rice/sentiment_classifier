from pathlib import Path
import joblib
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

STEMS = ["Reviews_tfidf", "Reviews_tfidf_overcleaned"]

# Focus on stable regularization strengths: C ranges from 0.01 to 10
alphas = [0.1, 1, 10, 100]
param_grid = {"C": [1 / a for a in alphas]}

for stem in STEMS:
    X_train = sp.load_npz(PROCESSED / f"{stem}_train_X.npz")
    X_test = sp.load_npz(PROCESSED / f"{stem}_test_X.npz")
    y_train = pd.read_csv(PROCESSED / f"{stem}_train_y.csv")["sentiment"].values.ravel()
    y_test = pd.read_csv(PROCESSED / f"{stem}_test_y.csv")["sentiment"].values.ravel()

    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)

    # Switched to solver="saga" and tol=1e-3 to prevent float overflow on large sparse inputs
    base_model = LogisticRegression(
        solver="saga",
        max_iter=1000,
        tol=1e-3,
        random_state=42,
    )

    search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,  # Parallelize grid search across CPU cores for faster execution
    )
    search.fit(X_train, y_train)

    best_alpha = 1 / search.best_params_["C"]
    report = classification_report(y_test, search.predict(X_test))

    print(f"\n=== {stem} ===")
    print("Best alpha:", best_alpha)
    print(report)

    (REPORTS / f"logreg_{stem}.txt").write_text(
        f"Best alpha: {best_alpha}\n\n{report}"
    )
    joblib.dump(search.best_estimator_, MODELS / f"logreg_{stem}.joblib")