"""Train the FINAL router on all outcome-labelled queries and save it to models/router.npz.

Input = the LLM's own confidence features only (5 numbers, see CONF_COLS in train_classifier.py).
Why this input: in cross-validation it matched the best text+confidence model
(macro-F1 0.436 vs 0.440, within noise) and it is the simplest: no encoder is needed at inference.

Note: the class weights make the probabilities favour rare classes; they are NOT calibrated.
(Calibration is the conformal step that is still to be done.)
"""
import os

import numpy as np

from common import DATA, load_rows
from train_classifier import CLASSES, CONF_COLS, confusion, f1_per_class, forward, train

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def main():
    rows = load_rows(os.path.join(DATA, "labels_relaxed.csv"))
    conf = {r["query"]: r for r in load_rows(os.path.join(DATA, "conf_A.csv"))}
    keep = [r for r in rows if r["label_source"] == "outcome"]

    X = np.array([[float(conf[r["query"]][c]) for c in CONF_COLS] for r in keep])
    y = np.array([CLASSES.index(r["label"]) for r in keep])
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd

    P = train(Xs, y, seed=0)

    pred = forward(P, Xs)[2].argmax(axis=1)
    C = confusion(y, pred)
    print(f"trained on {len(y)} outcome-labelled queries: "
          + ", ".join(f"{c}={int((y == i).sum())}" for i, c in enumerate(CLASSES)))
    print(f"TRAIN-set macro-F1 {f1_per_class(C).mean():.3f} (sanity check only, NOT a test score;"
          " the honest number is the cross-validation result)")

    os.makedirs(MODELS, exist_ok=True)
    path = os.path.join(MODELS, "router.npz")
    np.savez(path, W1=P["W1"], b1=P["b1"], W2=P["W2"], b2=P["b2"],
             mu=mu, sd=sd, classes=np.array(CLASSES))
    print("saved", os.path.normpath(path))


if __name__ == "__main__":
    main()
