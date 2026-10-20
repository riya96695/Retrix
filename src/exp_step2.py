"""Step 2b: which post-retrieval signals predict that the single-search (B) answer is correct?
Target: correct_B. Scratch logistic regression, 5-fold x 5 seeds, AUROC.
Run: python src/exp_step2.py
"""
import csv
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

GROUPS = {
    "1 retrieval score": ["bm25_top1", "bm25_gap", "bm25_mean", "bm25_top1_per_tok"],
    "2 chunk agreement": ["agree_mean", "agree_min"],
    "3 generation confidence": ["b_mean_lp", "b_min_lp", "b_first_lp", "b_mean_ent", "b_max_ent", "b_n_tok"],
    "4 query-chunk overlap": ["lex_all", "lex_top1", "sem_top1", "sem_max", "sem_mean"],
}
A_CONF = ["mean_lp", "min_lp", "first_lp", "first_ent", "n_tok"]  # Stage 1 (no-search) confidence


def read(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return {r["query"]: r for r in csv.DictReader(f)}


sig, b, a = read("signals_B.csv"), read("answers_B_all.csv"), read("conf_A.csv")
qs = [q for q in sig if q in b and q in a]
y = np.array([int(b[q]["correct_B"]) for q in qs])
print(f"{len(qs)} queries | B correct {y.mean():.3f}")


def cols(names, src):
    return np.nan_to_num(np.array([[float(src[q][n]) for n in names] for q in qs]))


def auroc(y, s):
    ranks = np.empty(len(s)); ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


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


def cv_auroc(X):
    res = []
    for seed in range(5):
        oof = np.zeros(len(y)); fs = folds(y, 5, seed)
        for i, te in enumerate(fs):
            tr = np.concatenate([f for j, f in enumerate(fs) if j != i])
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
            w, b0 = fit((X[tr] - mu) / sd, y[tr])
            oof[te] = ((X[te] - mu) / sd) @ w + b0
        res.append(auroc(y, oof))
    return np.mean(res), np.std(res)


Xg = {name: cols(c, sig) for name, c in GROUPS.items()}
Xa = cols(A_CONF, a)
print("\nAUROC for 'B answer is correct' (0.5 = random)")
for name, X in Xg.items():
    m, s = cv_auroc(X); print(f"{name:28s} {m:.3f} +- {s:.3f}")
m, s = cv_auroc(Xa); print(f"{'0 no-search confidence':28s} {m:.3f} +- {s:.3f}")
m, s = cv_auroc(np.hstack(list(Xg.values()))); print(f"{'all 4 groups':28s} {m:.3f} +- {s:.3f}")
m, s = cv_auroc(np.hstack(list(Xg.values()) + [Xa])); print(f"{'all 4 + no-search conf':28s} {m:.3f} +- {s:.3f}")