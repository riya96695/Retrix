"""3-class router v2: NO_RETRIEVAL / SIMPLE / COMPLEX. Which inputs help?
Labels: A correct (strict correct_A) -> NO; else HotpotQA -> COMPLEX; else SIMPLE.
Inputs: dataset name only | confidence only | query embedding only | confidence + query embedding.
Run: python src/exp_3class_v2.py
"""
import csv
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CONF = ["mean_lp", "min_lp", "first_lp", "first_ent", "n_tok"]


def read(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


a_rows = read("answers_A.csv")
conf = {r["query"]: r for r in read("conf_A.csv")}
lab = read("labels_relaxed.csv")
emb = np.load(os.path.join(DATA, "emb_query.npy"))
assert len(emb) == len(lab), "emb_query.npy and labels_relaxed.csv have different lengths"
qidx = {r["query"]: i for i, r in enumerate(lab)}

rows = [r for r in a_rows if r["query"] in conf and r["query"] in qidx]
y = np.array([0 if r["correct_A"] == "1" else (2 if "hotpot" in r["source_dataset"].lower() else 1)
              for r in rows])
C = np.nan_to_num(np.array([[float(conf[r["query"]][k]) for k in CONF] for r in rows]))
Q = np.array([emb[qidx[r["query"]]] for r in rows])
D = np.array([[0.0, 1.0] if "hotpot" in r["source_dataset"].lower() else [1.0, 0.0] for r in rows])
print(f"{len(y)} queries | NO {(y == 0).sum()} | SIMPLE {(y == 1).sum()} | COMPLEX {(y == 2).sum()}")


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train(Xtr, ytr, seed, hidden=32, epochs=300, lr=0.01, wd=1e-4):
    rng = np.random.default_rng(seed)
    d, k = Xtr.shape[1], 3
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


def predict(params, Xte):
    W1, b1, W2, b2 = params
    return softmax(np.maximum(0, Xte @ W1 + b1) @ W2 + b2).argmax(1)


def folds(y, k, seed):
    rng = np.random.default_rng(seed)
    idx = [[] for _ in range(k)]
    for c in np.unique(y):
        ids = np.where(y == c)[0]
        rng.shuffle(ids)
        for j, i in enumerate(ids):
            idx[j % k].append(i)
    return [np.array(f) for f in idx]


def f1s(y, p):
    out = []
    for c in range(3):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        out.append(2 * tp / max(2 * tp + fp + fn, 1))
    return np.array(out)


def run(X):
    macro, per, acc = [], [], []
    for seed in range(5):
        oof = np.zeros(len(y), dtype=int)
        fs = folds(y, 5, seed)
        for i, te in enumerate(fs):
            tr = np.concatenate([f for j, f in enumerate(fs) if j != i])
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
            params = train((X[tr] - mu) / sd, y[tr], seed)
            oof[te] = predict(params, (X[te] - mu) / sd)
        f = f1s(y, oof)
        per.append(f); macro.append(f.mean()); acc.append((oof == y).mean())
    return np.mean(macro), np.std(macro), np.mean(acc), np.mean(per, axis=0)


print(f"\n{'input':32s} {'macro-F1':>14} {'acc':>6} | {'F1 NO':>6} {'SIMPLE':>7} {'COMPLEX':>8}")
for name, X in (("dataset name only", D), ("confidence only", C),
                ("query embedding only", Q), ("confidence + query embedding", np.hstack([C, Q]))):
    m, s, a, p = run(X)
    print(f"{name:32s} {m:.3f} +- {s:.3f} {a:>6.3f} | {p[0]:>6.3f} {p[1]:>7.3f} {p[2]:>8.3f}")