"""Builds report.tex and compiles it to report.pdf with pdflatex.

Numbers are filled in from outputs/summary.json, outputs/validation.json and
outputs/cv_results_*.csv. Run after train.py, make_plots.py and validate.py.
Needs a LaTeX installation (pdflatex) with booktabs; otherwise compile
report.tex on Overleaf.
"""
import json
import shutil
import subprocess
from math import comb

import pandas as pd

S = {s["var"]: s for s in json.load(open("outputs/summary.json"))}
V = json.load(open("outputs/validation.json"))
cv = {v: pd.read_csv(f"outputs/cv_results_{v}.csv") for v in S}
s1, s2, v1, v2 = S["var1"], S["var2"], V["var1"], V["var2"]


def by_degree(v, degs):
    p = 6 if v == "var1" else 3
    rows = []
    for d in degs:
        g = cv[v][cv[v].degree == d].groupby("method").cv_mse.min()
        rows.append(f"{d} & {comb(d + p, p)} & {g['ridge']:.3f} & {g['lasso']:.3f} & {g['relaxed_lasso']:.3f} \\\\")
    return "\n".join(rows)


tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.1cm]{geometry}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{caption}
\usepackage{float}
\usepackage{parskip}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small}
\setlength{\parskip}{5pt}

\newcommand{\keybox}[1]{\par\noindent\fbox{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule}{\small #1}}\par}

\title{Polynomial Regression --- Assignment Report}
\author{Roll no.\ IMT2024067 \quad$\cdot$\quad Code: PyTorch\\[2pt] \small GitHub: \url{https://github.com/VishudhaSood/Assignment1_ML}}
\date{}

\begin{document}
\maketitle
\vspace{-1.5em}

\section{The task}
We get two datasets. In each one we must predict a number $y$ from some inputs, using
\textbf{only polynomial regression}. Each has 1000 training rows (with $y$) and 1000 test rows
(without $y$). The teacher scores our test predictions with \textbf{MSE} (lower is better) and
$\mathbf{R^2}$ (closer to 1 is better). The main decision is the \textbf{degree} of the polynomial.

\begin{table}[H]
\centering
\begin{tabular}{lll}
\toprule
 & var1 --- steam turbine & var2 --- thermal reservoir \\
\midrule
Inputs & 6 ($x_1 \ldots x_6$) & 3 ($x_1, x_2, x_3$) \\
Max degree allowed & 10 & 20 \\
Input values & between $-1$ and $1$ & between $-1$ and $1$ \\
$y$ in training data & from $-9.6$ to $12.4$ & from $-29.7$ to $30.2$ \\
\bottomrule
\end{tabular}
\end{table}

\textbf{What we noticed in the data.} No missing values. Many inputs are exactly $-1$ or $+1$
(the edge of the allowed range): about 30\% in var1 and 25\% in var2. The var1 \emph{test} file has
even more edge points than the training file, so we checked edge points separately (Section~5).

\section{Method}
\begin{center}
\small
\fbox{\parbox{2.3cm}{\centering 1.\ Make polynomial terms}} $\rightarrow$
\fbox{\parbox{2.3cm}{\centering 2.\ Fit with a penalty (Ridge / Lasso)}} $\rightarrow$
\fbox{\parbox{2.3cm}{\centering 3.\ Score with 5-fold CV}} $\rightarrow$
\fbox{\parbox{2.3cm}{\centering 4.\ Pick best degree + penalty}} $\rightarrow$
\fbox{\parbox{2.3cm}{\centering 5.\ Retrain on all data, predict test}}
\end{center}

\subsection{Polynomial features}
A polynomial of degree $d$ uses every term whose powers add up to at most $d$. For example with
two inputs and $d = 2$ the terms are: $1,\ x_1,\ x_2,\ x_1^2,\ x_1x_2,\ x_2^2$. The model is just a
weighted sum of these terms, so it is still \textbf{linear regression}, only on more columns. The
number of terms grows fast:

\begin{table}[H]
\centering
\begin{tabular}{lrrrrrr}
\toprule
Degree & 1 & 3 & 5 & 8 & 10 & 20 \\
\midrule
var1 (6 inputs) & 7 & 84 & 462 & 3003 & 8008 & --- \\
var2 (3 inputs) & 4 & 20 & 56 & 165 & 286 & 1771 \\
\bottomrule
\end{tabular}
\end{table}

With only 1000 rows, a high degree has more terms than rows. Plain least squares would then fit
the training data perfectly, noise included --- this is \textbf{overfitting}. A too-low degree misses
the real shape --- \textbf{underfitting}.

\keybox{\textbf{Small extra: Legendre polynomials.} Instead of raw powers $x, x^2, x^3, \ldots$ we build
the terms from Legendre polynomials $P_1(x), P_2(x), P_3(x), \ldots$ (e.g.\ $P_2 = (3x^2 - 1)/2$). They
describe \textbf{exactly the same set of polynomials}, so the model is unchanged. We use them because on
$[-1, 1]$ high powers like $x^{18}$ and $x^{20}$ look almost identical, which makes the computer's
equation solving unstable; Legendre terms do not have this problem.}

\subsection{Fitting with a penalty: Ridge and Lasso}
To stop overfitting we add a penalty on the size of the weights $w$ (regularisation):

\begin{table}[H]
\centering
\begin{tabularx}{\linewidth}{llX}
\toprule
Method & What we minimise & Effect \\
\midrule
Ridge (L2) & error $+ \lambda \sum w^2$ & shrinks all weights, keeps every term \\
Lasso (L1) & error $+ \alpha \sum |w|$ & sets useless weights to exactly 0 $\rightarrow$ keeps only some terms \\
Relaxed Lasso & Lasso first, then refit & Lasso chooses the terms; we then refit only those terms with
(almost) no penalty, so they are not shrunk too much \\
\bottomrule
\end{tabularx}
\end{table}

$\lambda$ and $\alpha$ control how strong the penalty is. Ridge has a direct formula,
\[ w = (\Phi^\top\Phi + \lambda n I)^{-1}\Phi^\top y, \]
solved with \texttt{torch.linalg.solve}. Lasso has no formula, so it is solved step by step
(gradient steps + pushing small weights to 0). The constant term (intercept) is never penalised.

\subsection{Choosing degree and penalty: 5-fold cross-validation}
We split the training rows into 5 equal parts. We train on 4 parts and measure the MSE on the
5th part, which the model has not seen, and repeat so every part is the `unseen' part once.
The average is the \textbf{CV MSE}, our estimate of the test error. We computed it for
\textbf{every degree} (1--10 for var1, 1--20 for var2), each method, and several penalty strengths,
and picked the lowest. That model was then retrained on all 1000 rows to predict the test file.

\section{var1 --- steam turbine (6 inputs)}
\begin{figure}[H]
\centering
\includegraphics[width=0.7\linewidth]{plots/cv_var1.png}
\caption{CV MSE vs degree (best penalty for each method). Lower is better.}
\end{figure}

\begin{table}[H]
\centering
\begin{tabular}{rrrrr}
\toprule
Degree & No.\ of terms & Ridge & Lasso & Relaxed Lasso \\
\midrule
""" + by_degree("var1", range(2, 11)) + r"""
\bottomrule
\end{tabular}
\caption{CV MSE for each degree and method.}
\end{table}

\keybox{\textbf{Chosen: degree """ + f"{s1['degree']}, relaxed Lasso ($\\alpha = {s1['strength']}$).}} " + \
    f"CV MSE $= {s1['cv_mse']:.3f}$, CV $R^2 = {s1['cv_r2']:.3f}$. Only {s1['n_terms_used']} of the " \
    f"{s1['n_terms_total']} possible terms are used." + r"""}

\textbf{Why degree 5?} From degree 1 to 5 the error drops a lot (lower degrees underfit). After 5 it
stops improving, so degree 5 is the simplest model that captures the pattern.

\textbf{Why Lasso and not Ridge?} At degree 5 Ridge's error is 0.55 but Lasso's is about 0.35. Ridge
keeps all 462 terms, while Lasso throws most of them away, so the real formula only uses a few
terms. Ridge also gets much worse at higher degrees (more and more useless terms), while Lasso
stays flat because it just removes them.

\section{var2 --- thermal reservoir (3 inputs)}
\begin{figure}[H]
\centering
\includegraphics[width=0.7\linewidth]{plots/cv_var2.png}
\caption{CV MSE vs degree. The minimum is at degree 8.}
\end{figure}

\begin{table}[H]
\centering
\begin{tabular}{rrrrr}
\toprule
Degree & No.\ of terms & Ridge & Lasso & Relaxed Lasso \\
\midrule
""" + by_degree("var2", [4, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]) + r"""
\bottomrule
\end{tabular}
\caption{CV MSE for selected degrees.}
\end{table}

\keybox{\textbf{Chosen: degree """ + f"{s2['degree']}, Ridge ($\\lambda = {s2['strength']}$).}} " + \
    f"CV MSE $= {s2['cv_mse']:.3f}$, CV $R^2 = {s2['cv_r2']:.3f}$. All {s2['n_terms_total']} terms are used." + r"""}

\textbf{Why degree 8 and not 20?} Degree 20 is allowed, but the CV error is lowest at 8 and rises
after that. Above 8 the extra terms mainly fit noise (overfitting). Degree 20 Ridge is about
$5\times$ worse than degree 8.

\textbf{Why Ridge?} At degree 8 all three methods give almost the same error, so there is no sign
that terms should be removed. We keep the simplest method, Ridge, which has a direct formula.

\section{How good are the predictions?}
The real test answers are hidden, so we estimated test performance in three ways:

\begin{table}[H]
\centering
\begin{tabularx}{\linewidth}{Xll}
\toprule
Check & var1 & var2 \\
\midrule
""" + f"""\\textbf{{Hold-out test}}: train on 80\\% of rows, test on the other 20\\% (10 random splits)
 & MSE {v1['holdout_mse']:.3f}, $R^2$ {v1['holdout_r2']:.3f} & MSE {v2['holdout_mse']:.3f}, $R^2$ {v2['holdout_r2']:.3f} \\\\
\\textbf{{Repeated 5-fold CV}} (3 different shuffles)
 & MSE {v1['repeated_cv_mse']:.3f}, $R^2$ {v1['repeated_cv_r2']:.3f} & MSE {v2['repeated_cv_mse']:.3f}, $R^2$ {v2['repeated_cv_r2']:.3f} \\\\
\\textbf{{Test-like MSE}}: CV error re-weighted so the share of edge points matches the test file
 & MSE $\\approx$ {v1['test_like_mse']:.2f} & MSE $\\approx$ {v2['test_like_mse']:.2f} \\\\
Baseline: always predict the average $y$ & MSE 9.99, $R^2$ 0 & MSE 42.1, $R^2$ 0 \\\\
""" + r"""\bottomrule
\end{tabularx}
\end{table}

All estimates agree, so the results are stable. We expect a test $R^2$ of about \textbf{0.96} for var1
and \textbf{0.99} for var2. The var1 test MSE may be slightly higher than the CV value ($\approx$0.38)
because its test file has more edge points, where errors are a bit larger
""" + f"({v1['mse_on_edge_rows']:.2f} on edge rows vs {v1['mse_on_inner_rows']:.2f} on inner rows)." + r"""

\begin{figure}[H]
\centering
\includegraphics[width=0.78\linewidth]{plots/parity.png}
\caption{Predicted vs actual $y$ on unseen CV rows. Points lie close to the diagonal.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.78\linewidth]{plots/residuals.png}
\caption{Errors (predicted $-$ actual). They are spread randomly around 0 with no pattern, so the
model is not missing any systematic shape.}
\end{figure}

\textbf{Other checks.} Both prediction files have 1000 rows, one column $y$, no missing values and
the same row order as the test files. Re-running the code reproduces them exactly.

\section{Summary}
\begin{table}[H]
\centering
\begin{tabular}{lll}
\toprule
 & var1 & var2 \\
\midrule
""" + f"""Degree & {s1['degree']} & {s2['degree']} \\\\
Method & Relaxed Lasso, $\\alpha = {s1['strength']}$ & Ridge, $\\lambda = {s2['strength']}$ \\\\
Terms used & {s1['n_terms_used']} of {s1['n_terms_total']} & {s2['n_terms_used']} of {s2['n_terms_total']} \\\\
CV MSE / $R^2$ & {s1['cv_mse']:.3f} / {s1['cv_r2']:.3f} & {s2['cv_mse']:.3f} / {s2['cv_r2']:.3f} \\\\
Hold-out MSE / $R^2$ & {v1['holdout_mse']:.3f} / {v1['holdout_r2']:.3f} & {v2['holdout_mse']:.3f} / {v2['holdout_r2']:.3f} \\\\
Prediction file & \\texttt{{IMT2024067\\_pred\\_var1.csv}} & \\texttt{{IMT2024067\\_pred\\_var2.csv}} \\\\
""" + r"""\bottomrule
\end{tabular}
\end{table}

\textbf{Main takeaways.} (1) The best degree is found by cross-validation, not by taking the maximum
allowed: 5 out of 10 for var1 and 8 out of 20 for var2. (2) Regularisation is what makes
high-degree polynomials usable; Lasso helps most when only a few terms matter (var1).
(3) Hold-out and CV estimates agree, so we expect similar errors on the hidden test set.

\textbf{Code} (GitHub: \url{https://github.com/VishudhaSood/Assignment1_ML}): \texttt{polyreg.py} model, \texttt{train.py} degree search + predictions,
\texttt{validate.py} checks, \texttt{make\_plots.py}, \texttt{make\_report.py}.

\end{document}
"""

with open("report.tex", "w") as f:
    f.write(tex)
print("report.tex written")

if shutil.which("pdflatex"):
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                       check=True, stdout=subprocess.DEVNULL)
    for ext in ("aux", "log", "out"):
        try:
            import os
            os.remove(f"report.{ext}")
        except FileNotFoundError:
            pass
    print("report.pdf written")
else:
    print("pdflatex not found: compile report.tex yourself (e.g. on Overleaf)")
