"""
Validation / diagnostics for the final models.

The real test labels are hidden, so we estimate test performance in three ways:
  1. Hold-out test : train on 80% of the training rows, score on the other 20%
                     (repeated over 10 random splits).
  2. Repeated 5-fold CV : 5-fold cross-validation repeated with 3 different shuffles.
  3. Edge check    : many test points lie on the edge of the input box (x = +-1).
                     We measure the error on training points that are on the edge
                     too, and re-weight the CV error so its mix of edge/non-edge
                     points matches the test file ("test-like MSE").
It also checks the prediction files (row count, NaNs, value range) and that
re-running the final model reproduces them exactly.

Usage:  python validate.py      (writes outputs/validation.json and plots/residuals.png)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from polyreg import make_model, mse, r2

ROLL = "IMT2024067"
FINAL = {  # chosen in train.py by cross-validation
    "var1": dict(degree=5, method="relaxed_lasso", strength=0.01),
    "var2": dict(degree=8, method="ridge", strength=0.001),
}


def load(var):
    tr = pd.read_csv(f"data/{ROLL}_train_{var}.csv")
    te = pd.read_csv(f"data/{ROLL}_test_{var}.csv")
    X = torch.tensor(tr.filter(like="x").values)
    y = torch.tensor(tr["y"].values)
    Xt = torch.tensor(te.values)
    return X, y, Xt


def n_edges(X):
    """How many inputs of each row sit exactly on the edge (+-1)."""
    return (X.abs() == 1).sum(1)


def edge_weights(X, Xt):
    """Weight each training row so the edge-count mix matches the test file."""
    k, kt = n_edges(X), n_edges(Xt)
    top = int(k.max())                     # test may have more edges than any train row
    k, kt = k.clamp(max=top), kt.clamp(max=top)
    w = torch.zeros(len(k))
    for b in range(top + 1):
        if (k == b).any():
            w[k == b] = ((kt == b).float().mean() / (k == b).float().mean()).item()
    return w


def fit_predict(cfg, X, y, Xn):
    m = make_model(X.shape[1], cfg["degree"], cfg["method"], cfg["strength"]).fit(X, y)
    with torch.no_grad():
        return m(Xn)


def holdout(cfg, X, y, repeats=10):
    mses, r2s = [], []
    for s in range(repeats):
        g = torch.Generator().manual_seed(100 + s)
        perm = torch.randperm(len(y), generator=g)
        te, tr = perm[:200], perm[200:]
        p = fit_predict(cfg, X[tr], y[tr], X[te])
        mses.append(mse(p, y[te]))
        r2s.append(r2(y[te], p))
    return sum(mses) / repeats, sum(r2s) / repeats


def repeated_cv(cfg, X, y, W, seeds=(0, 1, 2), k=5):
    oof_all, err, werr = [], [], []
    for s in seeds:
        g = torch.Generator().manual_seed(s)
        folds = torch.randperm(len(y), generator=g).chunk(k)
        oof = torch.zeros_like(y)
        for i in range(k):
            va = folds[i]
            tr = torch.cat([folds[j] for j in range(k) if j != i])
            oof[va] = fit_predict(cfg, X[tr], y[tr], X[va])
        e = (oof - y) ** 2
        err.append(e.mean().item())
        werr.append(((e * W).sum() / W.sum()).item())
        oof_all.append(oof)
    return oof_all[0], sum(err) / len(err), sum(werr) / len(werr)


def main():
    report = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (var, cfg) in zip(axes, FINAL.items()):
        X, y, Xt = load(var)
        W = edge_weights(X, Xt)
        ho_mse, ho_r2 = holdout(cfg, X, y)
        oof, cv_mse, test_like = repeated_cv(cfg, X, y, W)
        cv_r2 = 1 - cv_mse / torch.var(y, unbiased=False).item()

        k = n_edges(X)
        e = (oof - y) ** 2
        edge_rows = k >= max(2, X.shape[1] // 2)

        saved = torch.tensor(pd.read_csv(f"outputs/{ROLL}_pred_{var}.csv")["y"].values)
        again = fit_predict(cfg, X, y, Xt)

        report[var] = dict(
            model=cfg,
            holdout_mse=ho_mse, holdout_r2=ho_r2,
            repeated_cv_mse=cv_mse, repeated_cv_r2=cv_r2,
            test_like_mse=test_like,
            mse_on_edge_rows=e[edge_rows].mean().item(),
            mse_on_inner_rows=e[~edge_rows].mean().item(),
            pred_file_rows=len(saved), pred_file_nans=int(torch.isnan(saved).sum()),
            pred_range=[saved.min().item(), saved.max().item()],
            train_y_range=[y.min().item(), y.max().item()],
            reproduced_exactly=bool(torch.allclose(saved, again, atol=1e-8)),
        )

        ax.scatter(oof, oof - y, s=6, alpha=0.5, color="k")
        ax.axhline(0, color="0.5", lw=1)
        ax.set_title(f"{var}: residuals (out-of-fold)")
        ax.set_xlabel("predicted y")
        ax.set_ylabel("predicted - actual")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("plots/residuals.png", dpi=150)
    with open("outputs/validation.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
