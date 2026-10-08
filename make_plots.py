"""Extra report figures: out-of-fold predicted vs actual for the chosen models."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from polyreg import make_model
from train import ROLL, K, folds_for

S = json.load(open("outputs/summary.json"))
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, s in zip(axes, S):
    d = pd.read_csv(f"data/{ROLL}_train_{s['var']}.csv")
    X = torch.tensor(d.filter(like="x").values); y = torch.tensor(d.y.values)
    oof = torch.zeros_like(y)
    folds = folds_for(len(y))
    with torch.no_grad():
        for i in range(K):
            va = folds[i]; tr = torch.cat([folds[j] for j in range(K) if j != i])
            m = make_model(X.shape[1], s["degree"], s["method"], s["strength"]).fit(X[tr], y[tr])
            oof[va] = m(X[va])
    ax.scatter(y, oof, s=6, alpha=0.5, color="k")
    lo, hi = y.min().item(), y.max().item()
    ax.plot([lo, hi], [lo, hi], "--", color="0.5", lw=1)
    ax.set_title(f"{s['var']}: degree {s['degree']}, {s['method'].replace('_', ' ')}")
    ax.set_xlabel("actual y"); ax.set_ylabel("out-of-fold predicted y"); ax.grid(alpha=0.3)
    ax.text(0.04, 0.92, f"CV R² = {s['cv_r2']:.3f}", transform=ax.transAxes)
fig.tight_layout(); fig.savefig("plots/parity.png", dpi=160)
