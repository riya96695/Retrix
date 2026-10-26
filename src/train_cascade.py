"""Fit the final cascade router on all 1200 queries and save it to models/cascade.npz.
Gate 1: NO vs retrieve from the confidence features only.
Gate 2: SIMPLE vs COMPLEX from the query embedding, trained on non-NO queries only.
Labels: A correct (strict correct_A) -> NO; else HotpotQA -> COMPLEX; else SQuAD -> SIMPLE.
Run: python src/train_cascade.py
"""
import csv
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "models", "cascade.npz")
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
print(f"{len(y)} queries | NO {(y == 0).sum()} | SIMPLE {(y == 1).sum()} | COMPLEX {(y == 2).sum()}")


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train(Xtr, ytr, seed, k, hidden=32, epochs=300, lr=0.01, wd=1e-4):
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


def fit_gate(X, ylab, seed=0):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    W1, b1, W2, b2 = train((X - mu) / sd, ylab, seed, 2)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "mu": mu, "sd": sd}


g1 = fit_gate(C, (y != 0).astype(int))              # 0 = NO, 1 = retrieve
non = y != 0
g2 = fit_gate(Q[non], y[non] - 1)                   # 0 = SIMPLE, 1 = COMPLEX

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, conf_names=np.array(CONF),
         **{"g1_" + k: v for k, v in g1.items()}, **{"g2_" + k: v for k, v in g2.items()})
print("saved", OUT)

# sanity check: reload through the inference class and predict the training queries
from cascade import CascadeRouter

router = CascadeRouter()
IDX = {"NO_RETRIEVAL": 0, "SINGLE_STEP": 1, "MULTI_STEP": 2}
pred = np.array([IDX[router.predict(conf[r["query"]], Q[i])["route"]] for i, r in enumerate(rows)])
print(f"in-sample accuracy {(pred == y).mean():.3f} | predicted NO {(pred == 0).sum()} "
      f"SIMPLE {(pred == 1).sum()} COMPLEX {(pred == 2).sum()}")