import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
STEM = "IMDB_Dataset_tfidf"
SEED, TEST_SIZE = 42, 0.2
SKIP_MLP = "--skip-mlp" in sys.argv


def load_data():
    X_tr = sp.load_npz(PROCESSED / f"{STEM}_train_X.npz").tocsr()
    X_te = sp.load_npz(PROCESSED / f"{STEM}_test_X.npz").tocsr()
    y_tr = (pd.read_csv(PROCESSED / f"{STEM}_train_y.csv")["sentiment"] == "positive").astype(int).to_numpy()
    y_te = (pd.read_csv(PROCESSED / f"{STEM}_test_y.csv")["sentiment"] == "positive").astype(int).to_numpy()
    with open(PROCESSED / f"{STEM}_preprocessor.pkl", "rb") as f:
        vec = pickle.load(f)

    df = pd.read_csv(PROCESSED / "IMDB_Dataset.csv")
    df["review"] = df["review"].fillna("").astype(str)
    _, txt_te = train_test_split(df["review"], test_size=TEST_SIZE,
                                 random_state=SEED, stratify=df["sentiment"])
    assert ((df.loc[txt_te.index, "sentiment"] == "positive").astype(int).to_numpy() == y_te).all()
    return vec, X_tr, X_te, y_tr, y_te, txt_te


def evaluate(name, proba_fn, X_tr, y_tr, X_te, y_te, txt_te):
    p_tr, p_te = proba_fn(X_tr), proba_fn(X_te)
    pred = (p_te > 0.5).astype(int)
    text = (f"===== {name} =====\n"
            f"train accuracy: {((p_tr > 0.5) == y_tr).mean():.4f}   "
            f"test accuracy: {(pred == y_te).mean():.4f}\n\n"
            + classification_report(y_te, pred, target_names=["negative", "positive"], digits=4))
    print(text)
    (REPORTS / f"{name}_metrics.txt").write_text(text)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred),
                           display_labels=["negative", "positive"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} - confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(REPORTS / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    wrong = pd.DataFrame({"review": txt_te.to_numpy(), "true": y_te, "pred": pred, "p_positive": p_te})
    wrong = wrong[wrong.true != wrong.pred].copy()
    wrong["confidence"] = (wrong.p_positive - 0.5).abs()
    wrong.sort_values("confidence", ascending=False).head(50).to_csv(
        REPORTS / f"{name}_misclassified_examples.csv", index=False)


def train_logreg(X_tr, y_tr):
    grid = {"C": [1 / a for a in [0.01, 0.1, 1, 10, 100]]}
    search = GridSearchCV(LogisticRegression(max_iter=1000), grid, cv=5,
                          scoring="f1_macro", n_jobs=-1)
    print("Training Logistic Regression (a few minutes)...")
    search.fit(X_tr, y_tr)
    print("Best alpha:", 1 / search.best_params_["C"])
    return search.best_estimator_


def top_words(lr, vec, n=10):
    words = np.array(vec.get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    neg = pd.DataFrame({"word": words[order[:n]], "weight": lr.coef_[0][order[:n]]})
    pos = pd.DataFrame({"word": words[order[::-1][:n]], "weight": lr.coef_[0][order[::-1][:n]]})
    print(f"\nTop {n} most POSITIVE words:\n{pos.to_string(index=False)}")
    print(f"\nTop {n} most NEGATIVE words:\n{neg.to_string(index=False)}")
    return list(pos.word), list(neg.word)


def lr_sweeper(lr, X):
    X = sp.csc_matrix(X, copy=True)

    def sweep(col, grid):
        Xz = X.copy()
        Xz.data[Xz.indptr[col]:Xz.indptr[col + 1]] = 0
        Xz.eliminate_zeros()
        n = Xz.shape[0]
        out = []
        for v in grid:
            Xv = Xz if v == 0 else Xz + sp.csc_matrix(
                (np.full(n, v), (np.arange(n), np.full(n, col))), shape=Xz.shape)
            out.append(lr.predict_proba(Xv)[:, 1].mean())
        return np.array(out)
    return sweep


def mlp_sweeper(model, X, device):
    import torch
    W = model[0].weight.detach().cpu().numpy()
    base = X @ W.T + model[0].bias.detach().cpu().numpy()
    Xc = X.tocsc()
    rest = model[1:]

    def sweep(col, grid):
        xj = np.asarray(Xc[:, col].todense()).ravel()
        zero = base - np.outer(xj, W[:, col])
        out = []
        with torch.no_grad():
            for v in grid:
                h = torch.tensor(zero + v * W[:, col], dtype=torch.float32).to(device)
                out.append(torch.sigmoid(rest(h).squeeze(1)).mean().item())
        return np.array(out)
    return sweep


def plot_sweeps(sweepers, vec, X_te, words, grid):
    typical_max = np.percentile(X_te.data, 99)
    fig, axes = plt.subplots(1, len(sweepers), figsize=(6 * len(sweepers), 4.5),
                             sharey=True, squeeze=False)
    for ax, (name, sweep) in zip(axes[0], sweepers.items()):
        for w in words:
            ax.plot(grid, sweep(vec.vocabulary_[w], grid), marker="o", ms=3, label=w)
        ax.axvline(typical_max, color="gray", ls="--", lw=0.8)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_title(f"{name}: avg P(positive) as one word's value grows")
        ax.set_xlabel("TF-IDF value forced onto the word (in every review)")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("average P(positive) over test reviews")
    fig.tight_layout()
    fig.savefig(REPORTS / "word_sweeps.png", dpi=150)
    plt.close(fig)


def choose_mlp(mod, X_tr, y_tr, device):
    idx_fit, idx_val = train_test_split(np.arange(X_tr.shape[0]), test_size=0.1,
                                        random_state=SEED, stratify=y_tr)
    rows = []
    for n_layers in (1, 2, 3):
        rows += mod.train_and_validate(n_layers, X_tr[idx_fit], y_tr[idx_fit],
                                       X_tr[idx_val], y_tr[idx_val], device,
                                       num_epochs=5, seed=SEED)
    hist = pd.DataFrame(rows)
    hist.to_csv(REPORTS / "mlp_layer_search.csv", index=False)
    hist["score"] = hist["val_f1"].round(3)
    best = hist.sort_values(["score", "num_layers", "epoch"],
                            ascending=[False, True, True]).iloc[0]
    print(f"MLP chosen: {int(best['num_layers'])} hidden layer(s), {int(best['epoch'])} epoch(s)")
    return int(best["num_layers"]), int(best["epoch"])


def train_mlp(mod, X_tr, y_tr, layers, epochs, device):
    import torch
    from torch import nn
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = mod.build_model(X_tr.shape[1], layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(1, epochs + 1):
        print(f"  MLP epoch {ep}  train loss {mod.train_epoch(model, X_tr, y_tr, opt, loss_fn, device):.4f}")
    return model.eval()


def mlp_proba_fn(model, mod, device):
    import torch

    def proba(X):
        out = []
        with torch.no_grad():
            for s in range(0, X.shape[0], 512):
                scores = model(mod.to_tensor(X[s:s + 512], device)).squeeze(1)
                out.append(torch.sigmoid(scores).cpu().numpy())
        return np.concatenate(out)
    return proba


def main():
    REPORTS.mkdir(exist_ok=True)
    vec, X_tr, X_te, y_tr, y_te, txt_te = load_data()
    print(f"train {X_tr.shape}, test {X_te.shape}")

    lr = train_logreg(X_tr, y_tr)
    assert lr.coef_.shape[1] == X_te.shape[1] == len(vec.vocabulary_)
    evaluate("logreg", lambda X: lr.predict_proba(X)[:, 1], X_tr, y_tr, X_te, y_te, txt_te)
    pos, neg = top_words(lr, vec)
    words = pos[:3] + neg[:3]
    sweepers = {"logreg": lr_sweeper(lr, X_te)}

    if not SKIP_MLP:
        import torch
        sys.path.insert(0, str(MODELS))
        import multilayer_perceptron as mod
        device = "cuda" if torch.cuda.is_available() else "cpu"
        layers, epochs = choose_mlp(mod, X_tr, y_tr, device)
        model = train_mlp(mod, X_tr, y_tr, layers, epochs, device)
        evaluate("mlp", mlp_proba_fn(model, mod, device), X_tr, y_tr, X_te, y_te, txt_te)
        sweepers["mlp"] = mlp_sweeper(model, X_te, device)

    plot_sweeps(sweepers, vec, X_te, words, np.round(np.linspace(0, 3, 31), 2))
    print(f"\nOutputs written to {REPORTS}/")


if __name__ == "__main__":
    main()
