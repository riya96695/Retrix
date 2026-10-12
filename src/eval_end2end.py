"""Step 4: end-to-end evaluation of routing policies (numpy only, runs on a laptop).
Needs data/answers_B_all.csv and data/answers_C_all.csv (run gen_answers_B.py --all and gen_answers_C.py --all on a GPU).

For every query we know whether strategy A (no search), B (one search) and C (iterative search) answered correctly
(relaxed matcher) and how many searches each used. A policy picks one strategy per query:
  accuracy = fraction of queries answered correctly with the chosen strategy
  searches = average number of BM25 searches per query (A=0, B=1, C=number of rounds)
Policies: fixed strategies, the router (out-of-fold predictions, confidence features), the router used only to
skip search when P(NO_RETRIEVAL) >= t, and an oracle (cheapest strategy that is correct; counts as wrong if none is).
"""
import json
import os

import numpy as np

from common import DATA, load_rows, relaxed_correct
from train_classifier import CLASSES, CONF_COLS, FOLDS, SEEDS, forward, stratified_folds, train

THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def main():
    labels = load_rows(os.path.join(DATA, "labels_relaxed.csv"))
    A = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_A.csv"))}
    B = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_B_all.csv"))}
    C = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_C_all.csv"))}
    conf = {r["query"]: r for r in load_rows(os.path.join(DATA, "conf_A.csv"))}
    qs = [r["query"] for r in labels]
    missing = [q for q in qs if q not in B or q not in C]
    assert not missing, f"{len(missing)} queries have no B/C answer yet: run gen_answers_B.py --all and gen_answers_C.py --all"

    gold = {q: json.loads(A[q]["gold_answers"]) for q in qs}
    okA = np.array([relaxed_correct(A[q]["hypo_answer"], gold[q]) for q in qs])
    okB = np.array([relaxed_correct(B[q]["answer_B"], gold[q]) for q in qs])
    okC = np.array([relaxed_correct(C[q]["answer_C"], gold[q]) for q in qs])
    nC = np.array([int(C[q]["n_rounds"]) for q in qs])
    X = np.array([[float(conf[q][c]) for c in CONF_COLS] for q in qs])
    is_out = np.array([r["label_source"] == "outcome" for r in labels])
    y = np.array([CLASSES.index(r["label"]) if o else -1 for r, o in zip(labels, is_out)])
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

    out_idx, fb_idx = np.where(is_out)[0], np.where(~is_out)[0]
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        fold_o = stratified_folds(y[out_idx], FOLDS, seed)
        fold_f = rng.integers(0, FOLDS, len(fb_idx))      # fallback queries were never trained on
        probs = np.zeros((n, len(CLASSES)))
        for k in range(FOLDS):
            tr = out_idx[fold_o != k]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
            P = train((X[tr] - mu) / sd, y[tr], seed)
            te = np.concatenate([out_idx[fold_o == k], fb_idx[fold_f == k]])
            probs[te] = forward(P, (X[te] - mu) / sd)[2]
        rows.setdefault("router (argmax)", []).append(evaluate(probs.argmax(axis=1)))
        for t in THRESHOLDS:
            name = f"router: skip search if P(NO) >= {t}"
            rows.setdefault(name, []).append(evaluate(np.where(probs[:, 0] >= t, 0, 1)))

    print(f"{n} queries ({int(is_out.sum())} outcome-labelled, {len(fb_idx)} fallback); "
          f"router = 5-fold out-of-fold predictions x {len(SEEDS)} seeds, confidence features only")
    print(f"\n{'policy':40s}{'answer accuracy':>18s}{'avg searches':>16s}")
    for name, v in rows.items():
        acc, cost = np.array([a for a, _ in v]), np.array([c for _, c in v])
        print(f"{name:40s}{acc.mean():11.3f} +- {acc.std():.3f}{cost.mean():10.2f} +- {cost.std():.2f}")


if __name__ == "__main__":
    main()
