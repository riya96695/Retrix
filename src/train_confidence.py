"""Step 2d: fit the final confidence model on all 1200 queries and save it to models/confidence.npz.
Also exposes load()/score() so rag.py can import them.
Run: python src/train_confidence.py
"""
import csv
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data")
PATH = os.path.join(ROOT, "models", "confidence.npz")
FEATURES = ["bm25_top1", "bm25_gap", "bm25_mean", "bm25_top1_per_tok", "agree_mean", "agree_min",
            "b_mean_lp", "b_min_lp", "b_first_lp", "b_mean_ent", "b_max_ent", "b_n_tok",
            "lex_all", "lex_top1", "sem_top1", "sem_max", "sem_mean"]


def load():
    m = np.load(PATH)
    return {k: m[k] for k in ("mu", "sd", "w", "b")}


def score(feats, model=None):
    """feats: dict with all FEATURES for one query -> estimated P(single-search answer is correct)."""
    m = model or load()
    x = (np.array([feats[n] for n in FEATURES], dtype=float) - m["mu"]) / m["sd"]
    return float(1 / (1 + np.exp(-(x @ m["w"] + m["b"]))))


def main():
    def read(name):
        with open(os.path.join(DATA, name), encoding="utf-8") as f:
            return {r["query"]: r for r in csv.DictReader(f)}

    sig, b = read("signals_B.csv"), read("answers_B_all.csv")
    qs = [q for q in sig if q in b]
    y = np.array([int(b[q]["correct_B"]) for q in qs], dtype=float)
    X = np.nan_to_num(np.array([[float(sig[q][n]) for n in FEATURES] for q in qs]))
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    w, b0 = np.zeros(Z.shape[1]), 0.0
    for _ in range(500):
        p = 1 / (1 + np.exp(-(Z @ w + b0)))
        g = (p - y) / len(y)
        w -= 0.1 * (Z.T @ g + 1e-2 * w); b0 -= 0.1 * g.sum()
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    np.savez(PATH, mu=mu, sd=sd, w=w, b=b0, features=np.array(FEATURES))
    s = Z @ w + b0
    r = np.empty(len(s)); r[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    print(f"saved {PATH} | {len(y)} queries | in-sample AUROC {(r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0):.3f}")


if __name__ == "__main__":
    main()