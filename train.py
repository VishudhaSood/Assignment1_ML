"""
Degree / regularisation search + final training + test inference.

For each problem:
  1. 5-fold cross-validation over every degree 1..cap, for three ways of
     fitting the polynomial coefficients:
        ridge          (L2 penalty, closed form)
        lasso          (L1 penalty, FISTA)
        relaxed_lasso  (L1 to pick terms, then near-unpenalised refit)
  2. Pick the (degree, method, strength) with the lowest CV MSE.
  3. Refit on all training rows, predict the test file, save CSV.

Usage:  python train.py            (writes outputs/ and plots/)
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from polyreg import (PolynomialFeatures, PolynomialRegressor, SparsePolynomialRegressor,
                     make_model, mse, r2)

ROLL = "IMT2024067"
K = 5
SEED = 0

PROBLEMS = {
    "var1": dict(max_degree=10,
                 ridge=[0, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
                 lasso=[3e-2, 1e-2, 3e-3]),
    "var2": dict(max_degree=20,
                 ridge=[0, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
                 lasso=[3e-3, 1e-3, 3e-4]),
}


def folds_for(n):
    g = torch.Generator().manual_seed(SEED)
    return torch.randperm(n, generator=g).chunk(K)


def cv_all(X, y, degree, cfg):
    """CV MSE for every (method, strength) at one degree. Returns {(method, s): mse}."""
    folds = folds_for(len(y))
    res = {}
    for i in range(K):
        va = folds[i]
        tr = torch.cat([folds[j] for j in range(K) if j != i])
        for lam in cfg["ridge"]:
            m = PolynomialRegressor(X.shape[1], degree, lam).fit(X[tr], y[tr])
            res.setdefault(("ridge", lam), []).append(mse(m(X[va]), y[va]))
        # one Lasso path per fold (large -> small alpha, warm-started)
        lasso = SparsePolynomialRegressor(X.shape[1], degree, cfg["lasso"][0], max_iter=8000, tol=1e-8)
        for a in cfg["lasso"]:
            lasso.alpha, lasso.refit = a, False
            lasso.fit(X[tr], y[tr])
            res.setdefault(("lasso", a), []).append(mse(lasso(X[va]), y[va]))
            relaxed = SparsePolynomialRegressor(X.shape[1], degree, a, refit=True)
            relaxed.features = lasso.features
            relaxed.w0, relaxed.max_iter = lasso.w0, 1   # reuse the converged Lasso solution
            relaxed.fit(X[tr], y[tr])
            res.setdefault(("relaxed_lasso", a), []).append(mse(relaxed(X[va]), y[va]))
    return {k: sum(v) / len(v) for k, v in res.items()}


@torch.no_grad()
def run(var, cfg):
    tr = pd.read_csv(f"data/{ROLL}_train_{var}.csv")
    te = pd.read_csv(f"data/{ROLL}_test_{var}.csv")
    xcols = [c for c in tr.columns if c.startswith("x")]
    X = torch.tensor(tr[xcols].values)
    y = torch.tensor(tr["y"].values)
    Xt = torch.tensor(te[xcols].values)

    print(f"\n=== {var}: {len(xcols)} inputs, degree 1..{cfg['max_degree']} ===")
    table = []
    for d in range(1, cfg["max_degree"] + 1):
        t = time.time()
        res = cv_all(X, y, d, cfg)
        for (meth, s), v in res.items():
            table.append(dict(degree=d, method=meth, strength=s, cv_mse=v))
        best = min(res.items(), key=lambda kv: kv[1])
        nf = PolynomialFeatures(len(xcols), d).n_features
        print(f"deg {d:2d} | {nf:5d} terms | best {best[0][0]:13s} s={best[0][1]:<7g} "
              f"CV-MSE {best[1]:.4f} | {time.time() - t:.0f}s", flush=True)

    df = pd.DataFrame(table)
    df.to_csv(f"outputs/cv_results_{var}.csv", index=False)
    b = df.loc[df.cv_mse.idxmin()]
    deg, meth, s = int(b.degree), b.method, float(b.strength)
    var_y = torch.var(y, unbiased=False).item()
    print(f"--> chosen: degree {deg}, {meth}, strength {s}, CV-MSE {b.cv_mse:.4f}, "
          f"CV-R2 ~ {1 - b.cv_mse / var_y:.4f}")

    model = make_model(len(xcols), deg, meth, s).fit(X, y)
    yhat_tr = model(X)
    pred = model(Xt)
    pd.DataFrame({"y": pred.numpy()}).to_csv(f"outputs/{ROLL}_pred_{var}.csv", index=False)

    n_terms = model.features.n_features
    n_used = int((model.linear.weight.abs() > 0).sum())
    summary = dict(var=var, degree=deg, method=meth, strength=s,
                   cv_mse=float(b.cv_mse), cv_r2=1 - float(b.cv_mse) / var_y,
                   train_mse=mse(yhat_tr, y), train_r2=r2(y, yhat_tr),
                   n_terms_total=n_terms, n_terms_used=n_used,
                   y_train_range=[y.min().item(), y.max().item()],
                   y_pred_range=[pred.min().item(), pred.max().item()])
    plot_cv(df, var, deg, var_y)
    return summary


def plot_cv(df, var, chosen, var_y):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    style = {"ridge": ("k", "o", "-"), "lasso": ("0.35", "s", "--"), "relaxed_lasso": ("0.6", "^", ":")}
    for meth, g in df.groupby("method"):
        best = g.groupby("degree").cv_mse.min()
        c, mk, ls = style[meth]
        ax.plot(best.index, best.values, marker=mk, ls=ls, ms=4, label=meth.replace("_", " "), color=c)
    ax.axvline(chosen, ls="--", color="grey", lw=1)
    ax.text(chosen + 0.15, 0.93, f"chosen d={chosen}", color="grey", fontsize=9,
            transform=ax.get_xaxis_transform())
    ax.axhline(var_y, ls=":", color="#999", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("5-fold CV MSE (best strength, log scale)")
    ax.set_title(f"{var}: validation error vs degree")
    ax.set_xticks(sorted(df.degree.unique()))
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"plots/cv_{var}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    summaries = [run(v, c) for v, c in PROBLEMS.items()]
    with open("outputs/summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(json.dumps(summaries, indent=2))
