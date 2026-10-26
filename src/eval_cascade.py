"""End-to-end evaluation of the cascade router, same matcher and cost rule as eval_end2end.py.
Gate 1 (confidence only) and Gate 2 (query embedding) are scored out-of-fold (5-fold x 5 seeds).
Cascade policy: P(NO) >= t -> A (no search); else Gate 2 simple -> B, complex -> C.
Also: 'skip if P(NO) >= t else always B' to isolate what Gate 2 adds.
Run: python src/eval_cascade.py
"""
import json
import os

import numpy as np

from common import DATA, load_rows, relaxed_correct

CONF = ["mean_lp", "min_lp", "first_lp", "first_ent", "n_tok"]
THRESHOLDS = (0.5, 0.7, 0.8, 0.9)
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


def proba(params, X):
    W1, b1, W2, b2 = params
    return softmax(np.maximum(0, X @ W1 + b1) @ W2 + b2)


def gate(Xtr, ytr, Xte, seed):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return proba(train((Xtr - mu) / sd, ytr, seed), (Xte - mu) / sd)


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
emb = np.load(os.path.join(DATA, "emb_query.npy"))
assert len(emb) == len(labels), "emb_query.npy and labels_relaxed.csv have different lengths"
qs = [r["query"] for r in labels]
missing = [q for q in qs if q not in A or q not in B or q not in C or q not in conf]
assert not missing, f"{len(missing)} queries are missing in answers_A/B_all/C_all/conf_A"

gold = {q: json.loads(A[q]["gold_answers"]) for q in qs}
okA = np.array([relaxed_correct(A[q]["hypo_answer"], gold[q]) for q in qs])
okB = np.array([relaxed_correct(B[q]["answer_B"], gold[q]) for q in qs])
okC = np.array([relaxed_correct(C[q]["answer_C"], gold[q]) for q in qs])
nC = np.array([int(C[q]["n_rounds"]) for q in qs])
Xc = np.nan_to_num(np.array([[float(conf[q][k]) for k in CONF] for q in qs]))
Q = emb
# cascade training labels, same as train_cascade.py: NO if strict correct_A, else HotpotQA = COMPLEX, else SIMPLE
y = np.array([0 if A[q]["correct_A"] == "1" else (2 if "hotpot" in A[q]["source_dataset"].lower() else 1)
              for q in qs])
n = len(qs)


def evaluate(route):
    ok = np.where(route == 0, okA, np.where(route == 1, okB, okC))
    cost = np.where(route == 0, 0, np.where(route == 1, 1, nC))
    return ok.mean(), cost.mean()


rows = {}
for name, r in (("always NO  (A)", 0), ("always SINGLE (B)", 1), ("always MULTI (C)", 2)):
    rows[name] = [evaluate(np.full(n, r))]
any_ok = okA | okB | okC
o_cost = np.where(okA, 0, np.where(okB, 1, np.where(okC, nC, 0)))
rows["oracle (cheapest correct)"] = [(any_ok.mean(), o_cost.mean())]

for seed in SEEDS:
    p_no, p_cx = np.zeros(n), np.zeros(n)
    fs = folds(y, FOLDS, seed)
    for i, te in enumerate(fs):
        tr = np.concatenate([f for j, f in enumerate(fs) if j != i])
        p_no[te] = gate(Xc[tr], (y[tr] != 0).astype(int), Xc[te], seed)[:, 0]     # Gate 1: P(NO)
        non = tr[y[tr] != 0]
        p_cx[te] = gate(Q[non], y[non] - 1, Q[te], seed)[:, 1]                    # Gate 2: P(complex)
    for t in THRESHOLDS:
        rows.setdefault(f"cascade: P(NO) >= {t} -> A, else simple B / complex C", []).append(
            evaluate(np.where(p_no >= t, 0, np.where(p_cx >= 0.5, 2, 1))))
    for t in THRESHOLDS:
        rows.setdefault(f"skip if P(NO) >= {t}, else always B", []).append(
            evaluate(np.where(p_no >= t, 0, 1)))

print(f"{n} queries | cascade scored out-of-fold, 5-fold x 5 seeds | matcher: relaxed_correct")
print(f"\n{'policy':62s}{'answer accuracy':>18s}{'avg searches':>16s}")
for name, v in rows.items():
    acc, cost = np.array([a for a, _ in v]), np.array([c for _, c in v])
    print(f"{name:62s}{acc.mean():11.3f} +- {acc.std():.3f}{cost.mean():10.2f} +- {cost.std():.2f}")