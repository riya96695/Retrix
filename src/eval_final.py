"""Final pipeline: Gate 1 (P(NO) >= t -> answer without search) -> B -> Step 2 score -> retry the lowest f with C.
Gate 1 = confidence-only MLP (same as cascade Gate 1), Step 2 score = logistic regression on the 17 signals.
Everything out-of-fold, 5-fold x 5 seeds, relaxed matcher, cost: A = 0, B = 1, retry = 1 (B) + C rounds.
Run: python src/eval_final.py
"""
import json
import os

import numpy as np

from common import DATA, load_rows, relaxed_correct

CONF = ["mean_lp", "min_lp", "first_lp", "first_ent", "n_tok"]
FEATS = ["bm25_top1", "bm25_gap", "bm25_mean", "bm25_top1_per_tok", "agree_mean", "agree_min",
         "b_mean_lp", "b_min_lp", "b_first_lp", "b_mean_ent", "b_max_ent", "b_n_tok",
         "lex_all", "lex_top1", "sem_top1", "sem_max", "sem_mean"]
T_NO, RETRY = (0.8, 0.9), (0.1, 0.2, 0.3)
SEEDS, FOLDS = range(5), 5


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


labels = load_rows(os.path.join(DATA, "labels_relaxed.csv"))
A = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_A.csv"))}
B = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_B_all.csv"))}
C = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_C_all.csv"))}
conf = {r["query"]: r for r in load_rows(os.path.join(DATA, "conf_A.csv"))}
S = {r["query"]: r for r in load_rows(os.path.join(DATA, "signals_B.csv"))}
qs = [r["query"] for r in labels]
missing = [q for q in qs if q not in A or q not in B or q not in C or q not in conf or q not in S]
assert not missing, f"{len(missing)} queries missing in answers_A/B_all/C_all/conf_A/signals_B"

gold = {q: json.loads(A[q]["gold_answers"]) for q in qs}
okA = np.array([relaxed_correct(A[q]["hypo_answer"], gold[q]) for q in qs])
okB = np.array([relaxed_correct(B[q]["answer_B"], gold[q]) for q in qs])
okC = np.array([relaxed_correct(C[q]["answer_C"], gold[q]) for q in qs])
nC = np.array([int(C[q]["n_rounds"]) for q in qs])
Xc = np.nan_to_num(np.array([[float(conf[q][k]) for k in CONF] for q in qs]))
Xs = np.nan_to_num(np.array([[float(S[q][k]) for k in FEATS] for q in qs]))
yg = np.array([0 if A[q]["correct_A"] == "1" else 1 for q in qs])      # 0 = NO (strict), 1 = retrieve, as in train_cascade.py
yb = okB.astype(int)
n = len(qs)

rows = {}


def add(name, ok, cost):
    rows.setdefault(name, []).append((ok.mean(), cost.mean()))


for seed in SEEDS:
    p_no, s2 = np.zeros(n), np.zeros(n)
    fs = folds(yg, FOLDS, seed)
    for i, te in enumerate(fs):
        tr = np.concatenate([f for j, f in enumerate(fs) if j != i])
        p_no[te] = gate_p_no(Xc[tr], yg[tr], Xc[te], seed)
    fs = folds(yb, FOLDS, seed)
    for i, te in enumerate(fs):
        tr = np.concatenate([f for j, f in enumerate(fs) if j != i])
        mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-8
        w, b0 = fit((Xs[tr] - mu) / sd, yb[tr])
        s2[te] = ((Xs[te] - mu) / sd) @ w + b0
    rng = np.random.default_rng(seed)
    for t in T_NO:
        skip = p_no >= t
        kept = np.where(~skip)[0]
        base, base_cost = np.where(skip, okA, okB), np.where(skip, 0, 1)
        add(f"Gate 1 @{t}, else B", base, base_cost)
        for f in RETRY:
            k = int(round(f * len(kept)))
            retry = np.zeros(n, bool); retry[kept[np.argsort(s2[kept])[:k]]] = True
            add(f"Gate 1 @{t} + B + retry lowest {int(f * 100)}% -> C",
                np.where(retry, okC, base), np.where(retry, 1 + nC, base_cost))
            rr = np.zeros(n, bool); rr[rng.permutation(kept)[:k]] = True
            add(f"   same but RANDOM retry ({int(f * 100)}%, @{t})", np.where(rr, okC, base), np.where(rr, 1 + nC, base_cost))

print(f"{n} queries | everything out-of-fold, 5-fold x 5 seeds | matcher: relaxed_correct")
print(f"\n{'policy':58s}{'answer accuracy':>18s}{'avg searches':>16s}")
print(f"{'always B':58s}{okB.mean():11.3f}{'':7s}{1.0:10.2f}")
print(f"{'always C':58s}{okC.mean():11.3f}{'':7s}{nC.mean():10.2f}")
for name, v in rows.items():
    acc, cost = np.array([a for a, _ in v]), np.array([c for _, c in v])
    print(f"{name:58s}{acc.mean():11.3f} +- {acc.std():.3f}{cost.mean():10.2f} +- {cost.std():.2f}")