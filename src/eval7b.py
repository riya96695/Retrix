"""Evaluation of the 7B + hybrid run (data/run7b.csv). Accuracy uses the relaxed matcher (strict also printed).
Everything out-of-fold, 5-fold x 5 seeds. Gate 1 = confidence MLP (NO vs retrieve), Step 2 score = logistic regression
on the 17 signals. Cost: A = 0 searches, B = 1, retry = 1 (B) + C rounds. Run: python src/eval7b.py
"""
import json
import os

import numpy as np

from common import DATA, load_rows, relaxed_correct

CONF = ["mean_lp", "min_lp", "first_lp", "first_ent", "n_tok"]
FEATS = ["bm25_top1", "bm25_gap", "bm25_mean", "bm25_top1_per_tok", "agree_mean", "agree_min",
         "b_mean_lp", "b_min_lp", "b_first_lp", "b_mean_ent", "b_max_ent", "b_n_tok",
         "lex_all", "lex_top1", "sem_top1", "sem_max", "sem_mean"]
T_NO, RETRY, COVS = 0.9, (0.1, 0.2, 0.3), (1.0, 0.5, 0.3, 0.2, 0.1)

rows = load_rows(os.path.join(DATA, "run7b.csv"))
assert rows, "data/run7b.csv is missing or empty"
n = len(rows)
gold = [json.loads(r["gold_answers"]) for r in rows]
okA = np.array([relaxed_correct(r["ans_A"], g) for r, g in zip(rows, gold)])
okB = np.array([relaxed_correct(r["ans_B"], g) for r, g in zip(rows, gold)])
okC = np.array([relaxed_correct(r["ans_C"], g) for r, g in zip(rows, gold)])
nC = np.array([int(r["n_rounds"]) for r in rows])
sA = np.array([int(r["correct_A"]) for r in rows])
Xc = np.nan_to_num(np.array([[float(r[k]) for k in CONF] for r in rows]))
Xs = np.nan_to_num(np.array([[float(r[k]) for k in FEATS] for r in rows]))
yg = 1 - sA                       # 0 = NO (A strictly correct), 1 = retrieve
yb = okB.astype(int)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train(Xtr, ytr, seed, k=2, hidden=32, epochs=300, lr=0.01, wd=1e-4):
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    W1 = rng.normal(0, np.sqrt(2 / d), (d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, k)); b2 = np.zeros(k)
    cw = len(ytr) / (k * np.maximum(np.bincount(ytr, minlength=k).astype(float), 1))
    Y = np.eye(k)[ytr]
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    for t in range(1, epochs + 1):
        h = np.maximum(0, Xtr @ W1 + b1)
        p = softmax(h @ W2 + b2)
        dz2 = (p - Y) * cw[ytr][:, None] / len(ytr)
        dh = dz2 @ W2.T
        dh[h <= 0] = 0
        grads = [Xtr.T @ dh + wd * W1, dh.sum(0), h.T @ dz2 + wd * W2, dz2.sum(0)]
        for i, (par, g) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            par -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return params


def gate_p_no(Xtr, ytr, Xte, seed):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    W1, b1, W2, b2 = train((Xtr - mu) / sd, ytr, seed)
    return softmax(np.maximum(0, ((Xte - mu) / sd) @ W1 + b1) @ W2 + b2)[:, 0]


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
    for c in np.unique(y):
        ids = np.where(y == c)[0]
        rng.shuffle(ids)
        for j, i in enumerate(ids):
            idx[j % k].append(i)
    return [np.array(f) for f in idx]


def auroc(y, s):
    ranks = np.empty(len(s)); ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


res, au, rc, gate_stats = {}, [], {c: [] for c in COVS}, []


def add(name, ok, cost):
    res.setdefault(name, []).append((ok.mean(), cost.mean()))


for seed in range(5):
    p_no, s2 = np.zeros(n), np.zeros(n)
    fs = folds(yg, 5, seed)
    for i, te in enumerate(fs):
        tr = np.concatenate([f for j, f in enumerate(fs) if j != i])
        p_no[te] = gate_p_no(Xc[tr], yg[tr], Xc[te], seed)
    fs = folds(yb, 5, seed)
    for i, te in enumerate(fs):
        tr = np.concatenate([f for j, f in enumerate(fs) if j != i])
        mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-8
        w, b0 = fit((Xs[tr] - mu) / sd, yb[tr])
        s2[te] = ((Xs[te] - mu) / sd) @ w + b0
    au.append(auroc(yb, s2))
    for c in COVS:
        rc[c].append(okB[np.argsort(-s2)[:int(round(c * n))]].mean())
    rng = np.random.default_rng(seed)
    skip = p_no >= T_NO
    gate_stats.append((skip.mean(), okA[skip].mean() if skip.any() else float("nan")))
    for gate_on in (False, True):
        sk = skip if gate_on else np.zeros(n, bool)
        kept = np.where(~sk)[0]
        base, base_cost = np.where(sk, okA, okB), np.where(sk, 0, 1)
        tag = f"Gate 1 @{T_NO} + " if gate_on else ""
        if gate_on:
            add(f"Gate 1 @{T_NO}, else B", base, base_cost)
        for f in RETRY:
            k = int(round(f * len(kept)))
            retry = np.zeros(n, bool); retry[kept[np.argsort(s2[kept])[:k]]] = True
            add(f"{tag}B + retry lowest {int(f * 100)}% -> C", np.where(retry, okC, base), np.where(retry, 1 + nC, base_cost))
            rr = np.zeros(n, bool); rr[rng.permutation(kept)[:k]] = True
            add(f"   same, RANDOM retry {int(f * 100)}%", np.where(rr, okC, base), np.where(rr, 1 + nC, base_cost))

o_cost = np.where(okA, 0, np.where(okB, 1, np.where(okC, nC, 0)))
print(f"{n} queries | 7B + hybrid | out-of-fold, 5-fold x 5 seeds")
print(f"strict accuracy: A {sA.mean():.3f} | B {np.mean([int(r['correct_B']) for r in rows]):.3f} | "
      f"C {np.mean([int(r['correct_C']) for r in rows]):.3f}")
print(f"\n{'policy':58s}{'answer accuracy':>18s}{'avg searches':>16s}")
print(f"{'always A':58s}{okA.mean():11.3f}{'':7s}{0.0:10.2f}")
print(f"{'always B':58s}{okB.mean():11.3f}{'':7s}{1.0:10.2f}")
print(f"{'always C':58s}{okC.mean():11.3f}{'':7s}{nC.mean():10.2f}")
print(f"{'oracle (cheapest correct)':58s}{(okA | okB | okC).mean():11.3f}{'':7s}{o_cost.mean():10.2f}")
for name, v in res.items():
    a, c = np.array([x for x, _ in v]), np.array([y for _, y in v])
    print(f"{name:58s}{a.mean():11.3f} +- {a.std():.3f}{c.mean():10.2f} +- {c.std():.2f}")
sk = np.mean([s for s, _ in gate_stats]); sa = np.nanmean([a for _, a in gate_stats])
print(f"\nGate 1 @{T_NO}: skips {sk:.1%} of queries, A is correct on {sa:.1%} of them")
print(f"Step 2 score: AUROC for 'B is correct' = {np.mean(au):.3f}")
print("risk-coverage (accuracy of B on the most confident queries): " +
      " | ".join(f"{c:.0%}: {np.mean(rc[c]):.3f}" for c in COVS))