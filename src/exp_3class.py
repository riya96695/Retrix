"""3-class router: NO_RETRIEVAL / SIMPLE / COMPLEX, from LLM confidence features only.
Labels: A correct -> NO; else HotpotQA -> COMPLEX; else SIMPLE.
Run: python src/exp_3class.py
"""
import csv
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SKIP = ("query", "correct", "label", "gold", "answer", "id", "source", "route")


def read(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


a_rows = read("answers_A.csv")
conf = {r["query"]: r for r in read("conf_A.csv")}
cols = list(next(iter(conf.values())).keys())
feat = [c for c in cols
        if not any(s in c.lower() for s in SKIP) and all(num(r[c]) is not None for r in conf.values())]
print("features used:", feat)

X, y = [], []
for r in a_rows:
    c = conf.get(r["query"])
    if c is None:
        continue
    X.append([float(c[k]) for k in feat])
    if r["correct_A"] == "1":
        y.append(0)
    elif "hotpot" in r["source_dataset"].lower():
        y.append(2)
    else:
        y.append(1)
X = np.nan_to_num(np.array(X)); y = np.array(y)
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

per = np.mean(per, axis=0)
print(f"\nmacro-F1 {np.mean(macro):.3f} +- {np.std(macro):.3f} | accuracy {np.mean(acc):.3f}")
print(f"F1 per class: NO {per[0]:.3f} | SIMPLE {per[1]:.3f} | COMPLEX {per[2]:.3f}")