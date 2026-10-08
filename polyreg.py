# Polynomial regression in PyTorch.
# Polynomial = linear model on polynomial terms (nn.Linear on top of the features).
# Terms use Legendre polynomials (same polynomials as x, x^2, ... but more stable on [-1, 1]).
from itertools import combinations_with_replacement
from collections import Counter

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


def exponent_list(n_vars: int, degree: int) -> torch.Tensor:
    # all exponent tuples with sum <= degree (row 0 is the constant term)
    exps = []
    for d in range(degree + 1):
        for combo in combinations_with_replacement(range(n_vars), d):
            e = [0] * n_vars
            for v, c in Counter(combo).items():
                e[v] = c
            exps.append(e)
    return torch.tensor(exps, dtype=torch.long)


def legendre_table(x: torch.Tensor, degree: int) -> torch.Tensor:
    # Legendre polynomials P_0..P_degree for every column
    P = [torch.ones_like(x), x]
    for k in range(1, degree):
        # Bonnet recursion: (k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}
        P.append(((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1))
    return torch.stack(P[: degree + 1], dim=-1)


class PolynomialFeatures(nn.Module):
    def __init__(self, n_vars: int, degree: int):
        super().__init__()
        self.degree = degree
        self.register_buffer("exps", exponent_list(n_vars, degree))

    @property
    def n_features(self) -> int:
        return self.exps.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = legendre_table(x, self.degree)                     # (n, p, d+1)
        n, p, _ = T.shape
        # pick P_{a_j}(x_j) for every term and multiply across variables
        idx = self.exps.T.unsqueeze(0).expand(n, p, -1)        # (n, p, F)
        return torch.gather(T, 2, idx).prod(dim=1)             # (n, F)


class PolynomialRegressor(nn.Module):
    # ridge regression with a closed-form solution

    def __init__(self, n_vars: int, degree: int, lam: float = 0.0):
        super().__init__()
        self.features = PolynomialFeatures(n_vars, degree)
        self.linear = nn.Linear(self.features.n_features, 1, bias=False)
        self.lam = lam

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        Phi = self.features(X)
        n, F = Phi.shape
        reg = torch.ones(F)
        reg[0] = 0.0                                      # don't shrink the intercept
        lam = self.lam * n
        if F <= n:   # primal: (Phi^T Phi + lam R) w = Phi^T y
            A = Phi.T @ Phi + lam * torch.diag(reg) + 1e-10 * torch.eye(F)
            w = torch.linalg.solve(A, Phi.T @ y)
        else:        # dual (more terms than rows): w = Phi^T (Phi Phi^T + lam I)^-1 y
            mu = y.mean()                                  # centre y in place of an unpenalised bias
            K = Phi @ Phi.T
            a = torch.linalg.solve(K + (lam + 1e-10) * torch.eye(n), y - mu)
            w = Phi.T @ a
            w[0] += mu
        self.linear.weight.copy_(w.view(1, -1))
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(self.features(X)).squeeze(-1)


class SparsePolynomialRegressor(nn.Module):
    # Same model as above but fitted with an L1 (Lasso) penalty using FISTA.
    # relaxed Lasso (refit=True): Lasso picks the terms, then they are refit with a tiny ridge penalty.

    def __init__(self, n_vars, degree, alpha=1e-2, refit=False, max_iter=20000, tol=1e-9, standardize=False):
        super().__init__()
        self.features = PolynomialFeatures(n_vars, degree)
        F = self.features.n_features
        self.linear = nn.Linear(F, 1, bias=False)
        self.alpha, self.refit = alpha, refit
        self.max_iter, self.tol = max_iter, tol
        self.standardize = standardize
        self.w0 = None   # warm start

    @torch.no_grad()
    def fit(self, X, y):
        Phi = self.features(X)[:, 1:]                       # drop constant, handled as mean
        mu = Phi.mean(0)
        # Legendre columns are already O(1) on [-1,1]; leaving them unscaled means
        # the L1 penalty acts on the actual polynomial coefficients (works better here)
        sd = Phi.std(0).clamp_min(1e-12) if self.standardize else torch.ones(Phi.shape[1])
        Z = (Phi - mu) / sd
        ym = y.mean()
        yc = y - ym
        n, F = Z.shape
        L = torch.linalg.matrix_norm(Z, ord=2) ** 2 / n     # Lipschitz const of the gradient
        step = 1.0 / L
        w = torch.zeros(F) if self.w0 is None else self.w0.clone()
        v, t = w.clone(), 1.0
        for _ in range(self.max_iter):
            grad = Z.T @ (Z @ v - yc) / n
            u = v - step * grad
            w_new = torch.sign(u) * torch.clamp(u.abs() - step * self.alpha, min=0.0)  # soft-threshold
            t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
            v = w_new + ((t - 1) / t_new) * (w_new - w)
            if torch.max(torch.abs(w_new - w)) < self.tol:
                w = w_new
                break
            w, t = w_new, t_new
        self.w0 = w.clone()
        self.support = torch.nonzero(w).squeeze(-1)
        if self.refit and len(self.support) > 0:
            S = self.support
            Zs = Z[:, S]
            ws = torch.linalg.solve(Zs.T @ Zs + 1e-6 * n * torch.eye(len(S)), Zs.T @ yc)
            w = torch.zeros(F)
            w[S] = ws
        # fold standardisation back into one weight vector over the raw basis
        coef = w / sd
        bias = ym - (coef * mu).sum()
        self.linear.weight.copy_(torch.cat([bias.view(1), coef]).view(1, -1))
        return self

    def forward(self, X):
        return self.linear(self.features(X)).squeeze(-1)


def make_model(n_vars, degree, method, strength):
    if method == "ridge":
        return PolynomialRegressor(n_vars, degree, strength)
    if method == "lasso":
        return SparsePolynomialRegressor(n_vars, degree, strength)
    if method == "relaxed_lasso":
        return SparsePolynomialRegressor(n_vars, degree, strength, refit=True)
    raise ValueError(method)


def mse(a, b):
    return torch.mean((a - b) ** 2).item()


def r2(y, yhat):
    return 1.0 - torch.sum((y - yhat) ** 2).item() / torch.sum((y - y.mean()) ** 2).item()


def kfold_cv(X, y, degree, strength, method="ridge", k=5, seed=0):
    # mean validation MSE over k folds
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    folds = perm.chunk(k)
    errs = []
    for i in range(k):
        va = folds[i]
        tr = torch.cat([folds[j] for j in range(k) if j != i])
        m = make_model(X.shape[1], degree, method, strength).fit(X[tr], y[tr])
        errs.append(mse(m(X[va]), y[va]))
    return sum(errs) / k
