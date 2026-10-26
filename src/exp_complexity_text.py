"""Simple vs complex from the query WORDS only (unigrams + bigrams, scratch logistic regression).
Labels are dataset-defined: SQuAD = simple (0), HotpotQA = complex (1), all 1200 queries.
5-fold x 3 seeds. Also prints the words the model relies on. Run: python src/exp_complexity_text.py
"""
import csv
import os
import re

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
with open(os.path.join(DATA, "answers_A.csv"), encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
qs = [r["query"] for r in rows]
y = np.array([1 if "hotpot" in r["source_dataset"].lower() else 0 for r in rows])


def toks(q):
    w = re.findall(r"[a-z0-9']+", q.lower())
    return set(w + [a + "_" + b for a, b in zip(w, w[1:])])


docs = [toks(q) for q in qs]
print(f"{len(y)} queries | simple (SQuAD) {(y == 0).sum()} | complex (HotpotQA) {(y == 1).sum()}")


def vocab(idx, min_df=3, top=4000):
    df = {}
    for i in idx:
        for t in docs[i]:
            df[t] = df.get(t, 0) + 1
    keep = sorted((t for t, c in df.items() if c >= min_df), key=lambda t: -df[t])[:top]
    return {t: j for j, t in enumerate(keep)}


def matrix(idx, voc):
    X = np.zeros((len(idx), len(voc)))
    for r, i in enumerate(idx):
        for t in docs[i]:
            j = voc.get(t)
            if j is not None:
                X[r, j] = 1.0
    return X


def fit(X, yy, l2=1e-3, lr=0.5, epochs=500):
    w, b0 = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(X @ w + b0)))
        g = (p - yy) / len(yy)
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


acc, f1 = [], []
for seed in range(3):
    pred = np.zeros(len(y), dtype=int); fs = folds(y, 5, seed)
    for i, te in enumerate(fs):
        tr = np.concatenate([f for j, f in enumerate(fs) if j != i])
        voc = vocab(tr)
        w, b0 = fit(matrix(tr, voc), y[tr].astype(float))
        pred[te] = (matrix(te, voc) @ w + b0 > 0).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    acc.append((pred == y).mean()); f1.append(2 * tp / max(2 * tp + fp + fn, 1))
print(f"\naccuracy {np.mean(acc):.3f} +- {np.std(acc):.3f} | F1 complex {np.mean(f1):.3f}")

voc = vocab(np.arange(len(y)))
w, b0 = fit(matrix(np.arange(len(y)), voc), y.astype(float))
inv = {j: t for t, j in voc.items()}
order = np.argsort(w)
print("\nwords pushing towards SIMPLE :", ", ".join(inv[j] for j in order[:12]))
print("words pushing towards COMPLEX:", ", ".join(inv[j] for j in order[::-1][:12]))