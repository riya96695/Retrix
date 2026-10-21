"""Step 2c: risk-coverage. Answer only the top-X% queries by confidence score, report accuracy on those.
Scores are out-of-fold (5-fold x 5 seeds), so no query is scored by a model that saw it.
Run: python src/exp_risk_coverage.py
"""
import csv
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

GEN = ["b_mean_lp", "b_min_lp", "b_first_lp", "b_mean_ent", "b_max_ent", "b_n_tok"]
ALL = (["bm25_top1", "bm25_gap", "bm25_mean", "bm25_top1_per_tok", "agree_mean", "agree_min"]
       + GEN + ["lex_all", "lex_top1", "sem_top1", "sem_max", "sem_mean"])
COVS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


def read(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return {r["query"]: r for r in csv.DictReader(f)}


sig, b = read("signals_B.csv"), read("answers_B_all.csv")
qs = [q for q in sig if q in b]
y = np.array([int(b[q]["correct_B"]) for q in qs])
print(f"{len(qs)} queries | B correct {y.mean():.3f}")


def X_of(names):
    return np.nan_to_num(np.array([[float(sig[q][n]) for n in names] for q in qs]))


def fit(X, y, l2=1e-2, lr=0.1, epochs=300):
    w, b0 = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(X @ w + b0)))
        g = (p - y) / len(y)
        w -= lr * (X.T @ g + l2 * w); b0 -= lr * g.sum()
    return w, b0


def folds(y, k, seed):
    rng = np.random.default_rng(seed)
    idx = [[] for _ in range(k)]
    for c in (0, 1):
        ids = np.where(y == c)[0]; rng.shuffle(ids)
        for j, i in enumerate(ids):
            idx[j % k].append(i)
    return [np.array(f) for f in idx]


def oof_scores(X, seed):
    oof = np.zeros(len(y)); fs = folds(y, 5, seed)
    for i, te in enumerate(fs):
        tr = np.concatenate([f for j, f in enumerate(fs) if j != i])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        w, b0 = fit((X[tr] - mu) / sd, y[tr])
        oof[te] = ((X[te] - mu) / sd) @ w + b0
    return oof


def acc_at(s, cov):
    k = int(round(cov * len(y)))
    return y[np.argsort(-s)[:k]].mean()


res = {}
for name, names in (("generation confidence only", GEN), ("all 4 groups", ALL)):
    X = X_of(names)
    res[name] = [[acc_at(oof_scores(X, seed), c) for c in COVS] for seed in range(5)]

print(f"\n{'coverage':>9} {'answered':>9} {'gen-conf only':>14} {'all 4 groups':>13} {'random':>8}")
for j, c in enumerate(COVS):
    g = np.mean([r[j] for r in res["generation confidence only"]])
    a = np.mean([r[j] for r in res["all 4 groups"]])
    print(f"{c:>9.0%} {int(round(c * len(y))):>9d} {g:>14.3f} {a:>13.3f} {y.mean():>8.3f}")