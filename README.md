# Polynomial Regression Assignment - IMT2024067

Predicting `y` with **polynomial regression only**, for two datasets, written in **PyTorch**.

| Problem | Inputs | Chosen model | CV MSE | CV R² | Hold-out MSE / R² |
|---|---|---|---|---|---|
| var1 - steam turbine | 6 | degree 5, relaxed Lasso (α = 0.01), 105 of 462 terms | 0.353 | 0.965 | 0.352 / 0.964 |
| var2 - thermal reservoir | 3 | degree 8, Ridge (λ = 0.001), 165 terms | 0.301 | 0.993 | 0.296 / 0.993 |

Submission files: `outputs/IMT2024067_pred_var1.csv`, `outputs/IMT2024067_pred_var2.csv`
Report: `report.pdf`
Repository: https://github.com/VishudhaSood/Assignment1_ML

## How it works

1. **Polynomial features**: every term whose powers add up to ≤ degree (e.g. x1²·x3). Built from Legendre
   polynomials, which describe the same polynomials as x, x², x³… but are numerically more stable.
2. **Fit with a penalty** so that high degrees don't overfit:
   - Ridge (L2): shrinks all weights (direct formula, `torch.linalg.solve`)
   - Lasso (L1): sets useless weights to exactly 0 (solved iteratively)
   - Relaxed Lasso: Lasso picks the terms, then those terms are refit without shrinkage
3. **5-fold cross-validation** for every degree (1–10 for var1, 1–20 for var2), every method and several
   penalty strengths. The lowest CV MSE wins.
4. **Retrain** the winner on all training rows and predict the test file.
5. **Validate**: 80/20 hold-out test, repeated CV, edge-point check, prediction-file checks.

## Files

```
data/            given train/test CSVs
polyreg.py       the model: polynomial features + nn.Linear, Ridge and Lasso fitting, k-fold CV
train.py         degree / method / penalty search, final training, writes prediction CSVs
validate.py      hold-out test, repeated CV, edge check, checks the prediction files
make_plots.py    predicted-vs-actual plot
outputs/         predictions, all CV results (cv_results_*.csv), summary.json, validation.json
plots/           CV-vs-degree curves, predicted-vs-actual, residuals
report.pdf       write-up
```

## Run it

```bash
pip install -r requirements.txt
python train.py        # full search + predictions (~25 min on a laptop CPU; var1 high degrees are slow)
python validate.py     # ~1 min
python make_plots.py
```

Random seeds are fixed, so re-running gives the same prediction files.
