import ast
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "Amazon_Reviews_clean.csv"
TEXT_COL = "text_clean"
LABEL_COL = "label"
NB_ALPHA = 0.5

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
TRAIN_UP_PATH = ROOT / "data" / "processed" / "train_upsampled.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"


def read_settings():
    seed, test_size = 42, 0.2
    cfg_path = Path(__file__).resolve().with_name("config.yaml")
    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        seed = cfg.get("random_seed", seed)
        test_size = cfg.get("test_size", test_size)
    return seed, test_size


def load_data():
    df = pd.read_csv(DATA_PATH)
    for col in (TEXT_COL, LABEL_COL):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found in {DATA_PATH.name}. "
                             f"Columns are: {df.columns.tolist()}")
    df[TEXT_COL] = df[TEXT_COL].fillna("")
    if df[LABEL_COL].dtype == object:
        df[LABEL_COL] = df[LABEL_COL].str.lower().map({"positive": 1, "negative": 0})
    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def find_upsample_file():
    here = Path(__file__).resolve().parent / "upsample.py"
    if here.exists():
        return here
    found = [p for p in ROOT.rglob("upsample.py") if ".venv" not in p.parts]
    if not found:
        raise SystemExit("Could not find upsample.py.")
    return found[0]


def load_sunny_upsample():
    path = find_upsample_file()
    print(f"Using upsample() from {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    func = next((n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "upsample"), None)
    if func is None:
        raise SystemExit(f"No function called upsample() in {path.name}.")
    namespace = {"pd": pd}
    exec(compile(ast.Module([func], []), str(path), "exec"), namespace)
    return namespace["upsample"]


def upsample_train(train):
    upsample = load_sunny_upsample()
    train = train.copy()
    if "sentiment" not in train.columns:
        train["sentiment"] = train[LABEL_COL].map({1: "positive", 0: "negative"})
    out = upsample(train)
    return out.sample(frac=1, random_state=42).reset_index(drop=True)


def build_models(seed):
    return {
        "logreg": Pipeline([
            ("vec", TfidfVectorizer(max_features=50_000, min_df=5)),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "naive_bayes": Pipeline([
            ("vec", CountVectorizer(max_features=50_000, min_df=5)),
            ("clf", MultinomialNB(alpha=NB_ALPHA)),
        ]),
    }


def main():
    seed, test_size = read_settings()
    df = load_data()
    print(f"Rows: {len(df)}  (seed={seed}, test_size={test_size})")
    print("Overall class balance:")
    print(df[LABEL_COL].value_counts(normalize=True).round(3).to_string())

    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[LABEL_COL])
    print(f"\nTrain rows (before upsampling): {len(train)}")
    print(f"Test rows (kept as is): {len(test)}")

    train_up = upsample_train(train)
    print(f"Train rows after upsampling: {len(train_up)}")
    print("Train class balance after upsampling (should be 50/50):")
    print(train_up[LABEL_COL].value_counts(normalize=True).round(3).to_string())
    print("Test class balance (should be UNCHANGED):")
    print(test[LABEL_COL].value_counts(normalize=True).round(3).to_string())

    TRAIN_UP_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_up.to_csv(TRAIN_UP_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"\nSaved {TRAIN_UP_PATH.name} and {TEST_PATH.name} to data/processed/")

    X_train, y_train = train_up[TEXT_COL], train_up[LABEL_COL]
    X_test, y_test = test[TEXT_COL], test[LABEL_COL]

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    metric_rows, confusion_rows = [], []
    for name, pipe in build_models(seed).items():
        print(f"\n{'=' * 60}\nTraining {name}...\n{'=' * 60}")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        print(classification_report(y_test, preds,
                                    target_names=["negative", "positive"],
                                    digits=3))

        rep = classification_report(y_test, preds, output_dict=True)
        metric_rows.append({
            "model": name,
            "accuracy": rep["accuracy"],
            "neg_precision": rep["0"]["precision"],
            "neg_recall": rep["0"]["recall"],
            "neg_f1": rep["0"]["f1-score"],
            "pos_precision": rep["1"]["precision"],
            "pos_recall": rep["1"]["recall"],
            "pos_f1": rep["1"]["f1-score"],
            "macro_f1": rep["macro avg"]["f1-score"],
        })

        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        confusion_rows.append({"model": name, "true_neg": tn, "false_pos": fp,
                               "false_neg": fn, "true_pos": tp})
        print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

        out = MODELS_DIR / f"{name}_upsampled.joblib"
        joblib.dump(pipe, out)
        print(f"Saved model to {out}")

    pd.DataFrame(metric_rows).round(4).to_csv(
        REPORTS_DIR / "model_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(
        REPORTS_DIR / "confusion_matrices.csv", index=False)
    print("\nSaved reports/model_metrics.csv and reports/confusion_matrices.csv")


if __name__ == "__main__":
    main()
