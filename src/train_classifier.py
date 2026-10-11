"""Step 3b: query classifier head, written from scratch (numpy only).
 - 2-layer MLP, manual backprop, our own Adam, class-weighted cross-entropy, our own metrics
 - input: frozen pretrained embeddings from embed.py (fine-tuning the encoder comes later)
 - trains on outcome-labelled queries only (fallback labels are excluded)
Compares three inputs with identical splits/seeds:
   dataset name only | query only | query [SEP] answer (encoded together) | [query ; answer] (embedded separately, concatenated)
   + (if data/conf_A.csv exists) the LLM's own confidence alone, and [query ; answer ; confidence]
Stratified 5-fold cross-validation, repeated over several seeds (out-of-fold predictions).
Hyper-parameters are fixed in advance (not tuned on the test folds).
"""
import os

import numpy as np

from common import DATA, load_rows

CLASSES = ["NO_RETRIEVAL", "SINGLE_STEP", "MULTI_STEP"]
K = len(CLASSES)
HIDDEN, EPOCHS, LR, WD, BATCH = 128, 100, 1e-3, 1e-4, 64
FOLDS, SEEDS = 5, [0, 1, 2, 3, 4]
CONF_COLS = ["mean_lp", "min_lp", "first_lp", "first_ent", "n_tok"]   # from gen_conf.py


# ---------- model: 2-layer MLP, manual forward / backward ----------
def init_params(d_in, d_h, d_out, rng):
    return {"W1": rng.normal(0, np.sqrt(2 / d_in), (d_in, d_h)), "b1": np.zeros(d_h),
            "W2": rng.normal(0, np.sqrt(2 / d_h), (d_h, d_out)), "b2": np.zeros(d_out)}


def forward(P, X):
    z1 = X @ P["W1"] + P["b1"]
    h = np.maximum(z1, 0)
    z2 = h @ P["W2"] + P["b2"]
    z2 = z2 - z2.max(axis=1, keepdims=True)
    p = np.exp(z2)
    p /= p.sum(axis=1, keepdims=True)
    return z1, h, p


def loss_fn(P, X, y, cw):
    """Weighted cross-entropy + L2. Used by the gradient check."""
    _, _, p = forward(P, X)
    w = cw[y]
    nll = -np.log(p[np.arange(len(y)), y] + 1e-12)
    return (w * nll).sum() / w.sum() + 0.5 * WD * ((P["W1"] ** 2).sum() + (P["W2"] ** 2).sum())


def grads(P, X, y, cw):
    z1, h, p = forward(P, X)
    w = cw[y]
    dz2 = p.copy()
    dz2[np.arange(len(y)), y] -= 1
    dz2 *= (w / w.sum())[:, None]
    dz1 = (dz2 @ P["W2"].T) * (z1 > 0)
    return {"W2": h.T @ dz2 + WD * P["W2"], "b2": dz2.sum(0),
            "W1": X.T @ dz1 + WD * P["W1"], "b1": dz1.sum(0)}


class Adam:
    def __init__(self, P, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.P, self.lr, self.b1, self.b2, self.eps, self.t = P, lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}

    def step(self, G):
        self.t += 1
        for k in self.P:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * G[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * G[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.P[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train(X, y, seed):
    rng = np.random.default_rng(seed)
    cw = len(y) / (K * np.maximum(np.bincount(y, minlength=K), 1))   # class weights
    P = init_params(X.shape[1], HIDDEN, K, rng)
    opt = Adam(P, LR)
    for _ in range(EPOCHS):
        idx = rng.permutation(len(y))
        for i in range(0, len(y), BATCH):
            b = idx[i:i + BATCH]
            opt.step(grads(P, X[b], y[b], cw))
    return P


def predict(P, X):
    return forward(P, X)[2].argmax(axis=1)


# ---------- metrics (own implementation) ----------
def confusion(y, pred):
    C = np.zeros((K, K), dtype=int)
    for t, p in zip(y, pred):
        C[t, p] += 1
    return C


def f1_per_class(C):
    f1 = np.zeros(K)
    for c in range(K):
        tp, fp, fn = C[c, c], C[:, c].sum() - C[c, c], C[c, :].sum() - C[c, c]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1[c] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1


def stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for c in range(K):
        idx = rng.permutation(np.where(y == c)[0])
        fold[idx] = np.arange(len(idx)) % k
    return fold


def prep(X, n_std, tr, te):
    """Split into train/test; standardise the last n_std columns with TRAIN-fold statistics only."""
    Xtr, Xte = X[tr].copy(), X[te].copy()
    if n_std:
        mu, sd = Xtr[:, -n_std:].mean(0), Xtr[:, -n_std:].std(0) + 1e-8
        Xtr[:, -n_std:] = (Xtr[:, -n_std:] - mu) / sd
        Xte[:, -n_std:] = (Xte[:, -n_std:] - mu) / sd
    return Xtr, Xte


def dataset_only_predict(ds, y, tr, te):
    """Shortcut baseline: majority label of each source dataset in the training folds."""
    pred = np.zeros(te.sum(), dtype=int)
    for name in np.unique(ds):
        mask = tr & (ds == name)
        major = np.bincount(y[mask], minlength=K).argmax() if mask.any() else 0
        pred[ds[te] == name] = major
    return pred


def main():
    rows = load_rows(os.path.join(DATA, "labels_relaxed.csv"))
    Eq = np.load(os.path.join(DATA, "emb_query.npy"))
    Ep = np.load(os.path.join(DATA, "emb_pair.npy"))
    Ea = np.load(os.path.join(DATA, "emb_answer.npy"))
    assert len(rows) == len(Eq) == len(Ep) == len(Ea), "run embed.py again (row count mismatch)"

    keep = [i for i, r in enumerate(rows) if r["label_source"] == "outcome"]
    y = np.array([CLASSES.index(rows[i]["label"]) for i in keep])
    ds = np.array([rows[i]["source_dataset"] for i in keep])
    inputs = {"query only": (Eq[keep], 0),
              "query [SEP] answer": (Ep[keep], 0),
              "[query ; answer]": (np.hstack([Eq[keep], Ea[keep]]), 0)}

    conf_path = os.path.join(DATA, "conf_A.csv")
    if os.path.exists(conf_path):
        conf = {r["query"]: r for r in load_rows(conf_path)}
        Z = np.array([[float(conf[rows[i]["query"]][c]) for c in CONF_COLS] for i in keep])
        inputs["confidence only"] = (Z, len(CONF_COLS))
        inputs["[query ; answer ; conf]"] = (np.hstack([Eq[keep], Ea[keep], Z]), len(CONF_COLS))
    else:
        print("data/conf_A.csv not found -> confidence inputs skipped")

    print(f"outcome-labelled queries used: {len(y)}")
    for c, name in enumerate(CLASSES):
        print(f"  {name:13s} {int((y == c).sum()):4d}  "
              + "  ".join(f"{d}={int(((y == c) & (ds == d)).sum())}" for d in np.unique(ds)))

    methods = ["dataset name only"] + list(inputs)
    stats = {m: {"f1": [], "acc": [], "pc": [], "C": np.zeros((K, K), dtype=int)} for m in methods}

    for seed in SEEDS:
        fold = stratified_folds(y, FOLDS, seed)
        oof = {m: np.zeros(len(y), dtype=int) for m in methods}
        for k in range(FOLDS):
            tr, te = fold != k, fold == k
            oof["dataset name only"][te] = dataset_only_predict(ds, y, tr, te)
            for name, (X, n_std) in inputs.items():
                Xtr, Xte = prep(X, n_std, tr, te)
                oof[name][te] = predict(train(Xtr, y[tr], seed), Xte)
        for m in methods:
            C = confusion(y, oof[m])
            pc = f1_per_class(C)
            stats[m]["f1"].append(pc.mean())
            stats[m]["acc"].append(np.trace(C) / C.sum())
            stats[m]["pc"].append(pc)
            stats[m]["C"] += C

    print(f"\n{FOLDS}-fold CV x {len(SEEDS)} seeds (mean +- std over seeds)")
    print(f"{'method':26s}{'macro-F1':>16s}{'accuracy':>16s}   F1 per class (NO / SINGLE / MULTI)")
    for m in methods:
        s = stats[m]
        pc = np.mean(s["pc"], axis=0)
        print(f"{m:26s}{np.mean(s['f1']):9.3f} +- {np.std(s['f1']):.3f}"
              f"{np.mean(s['acc']):9.3f} +- {np.std(s['acc']):.3f}   "
              f"{pc[0]:.3f} / {pc[1]:.3f} / {pc[2]:.3f}")

    best = sorted(methods[1:], key=lambda m: -np.mean(stats[m]["f1"]))[:2]
    for m in best:
        print(f"\nconfusion matrix, {m} (rows = true, cols = predicted, summed over seeds)")
        print("               " + "".join(f"{c[:6]:>9s}" for c in CLASSES))
        for c, name in enumerate(CLASSES):
            print(f"{name:15s}" + "".join(f"{v:9d}" for v in stats[m]["C"][c]))


if __name__ == "__main__":
    main()
