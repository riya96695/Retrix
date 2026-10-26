"""Retry policy: answer with B (one search); if the Step 2 score says B is unsure, use C instead.
Score = logistic regression on the 17 Step 2 signals, out-of-fold (5-fold x 5 seeds), target = B correct (relaxed matcher).
Cost: kept queries use 1 search; retried queries use 1 (B) + C rounds.
Run: python src/eval_retry.py
"""
import json
import os

import numpy as np

from common import DATA, load_rows, relaxed_correct

FEATS = ["bm25_top1", "bm25_gap", "bm25_mean", "bm25_top1_per_tok", "agree_mean", "agree_min",
         "b_mean_lp", "b_min_lp", "b_first_lp", "b_mean_ent", "b_max_ent", "b_n_tok",
         "lex_all", "lex_top1", "sem_top1", "sem_max", "sem_mean"]
RETRY = (0.1, 0.2, 0.3, 0.4, 0.5)

labels = load_rows(os.path.join(DATA, "labels_relaxed.csv"))
A = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_A.csv"))}
B = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_B_all.csv"))}
C = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_C_all.csv"))}
S = {r["query"]: r for r in load_rows(os.path.join(DATA, "signals_B.csv"))}
qs = [r["query"] for r in labels]
missing = [q for q in qs if q not in A or q not in B or q not in C or q not in S]
assert not missing, f"{len(missing)} queries missing in answers_A / B_all / C_all / signals_B"

gold = {q: json.loads(A[q]["gold_answers"]) for q in qs}
okB = np.array([relaxed_correct(B[q]["answer_B"], gold[q]) for q in qs])
okC = np.array([relaxed_correct(C[q]["answer_C"], gold[q]) for q in qs])
nC = np.array([int(C[q]["n_rounds"]) for q in qs])
X = np.nan_to_num(np.array([[float(S[q][k]) for k in FEATS] for q in qs]))
y = okB.astype(int)
n = len(qs)


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


def policy(score, f):
    """Retry (use C) for the f fraction of queries with the LOWEST score."""
    retry = np.zeros(n, dtype=bool)
    retry[np.argsort(score)[:int(round(f * n))]] = True
    return np.where(retry, okC, okB).mean(), np.where(retry, 1 + nC, 1).mean()


res = {f: [] for f in RETRY}
for seed in range(5):
    oof = np.zeros(n); fs = folds(y, 5, seed)
    for i, te in enumerate(fs):
        tr = np.concatenate([f for j, f in enumerate(fs) if j != i])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        w, b0 = fit((X[tr] - mu) / sd, y[tr])
        oof[te] = ((X[te] - mu) / sd) @ w + b0
    for f in RETRY:
        res[f].append(policy(oof, f))

rng = np.random.default_rng(0)
print(f"{n} queries | matcher: relaxed_correct | retry cost = 1 (B) + C rounds")
print(f"always B: {okB.mean():.3f} @ 1.00 searches | always C: {okC.mean():.3f} @ {nC.mean():.2f} searches")
print(f"\n{'retry the least-sure':22s}{'Step 2 score':>22s}{'random retry (measured)':>34s}")
print(f"{'':22s}{'accuracy  searches':>22s}{'accuracy  searches':>34s}")
for f in RETRY:
    a = np.mean([x for x, _ in res[f]]); c = np.mean([x for _, x in res[f]])
    rr = [policy(rng.random(n), f) for _ in range(300)]
    print(f"{f:>21.0%} {a:>12.3f}{c:>10.2f}{np.mean([x for x, _ in rr]):>24.3f}{np.mean([x for _, x in rr]):>10.2f}")
ok_or = okB | okC
print(f"\nceiling, perfect retry (retry only when B is wrong): accuracy {ok_or.mean():.3f} "
      f"@ {np.where(okB, 1, 1 + nC).mean():.2f} searches")