import joblib
import scipy.sparse as sp
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

STEMS = ["IMDB_Dataset_tfidf", "IMDB_Dataset_tfidf_overcleaned"]

alphas = [0.01, 0.1, 1, 10, 100]
param_grid = {"C": [1 / a for a in alphas]}

for stem in STEMS:
    X_train = sp.load_npz(PROCESSED / f"{stem}_train_X.npz")
    X_test = sp.load_npz(PROCESSED / f"{stem}_test_X.npz")
    y_train = pd.read_csv(PROCESSED / f"{stem}_train_y.csv")["sentiment"]
    y_test = pd.read_csv(PROCESSED / f"{stem}_test_y.csv")["sentiment"]

    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)

    search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid, cv=5, scoring="f1_macro",
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