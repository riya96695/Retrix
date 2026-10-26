"""Cascade router for live use. Loads models/cascade.npz written by train_cascade.py.
Gate 1 (confidence features) decides NO vs retrieve; Gate 2 (query embedding) decides SIMPLE vs COMPLEX.
Route names match the old router: NO_RETRIEVAL / SINGLE_STEP (simple) / MULTI_STEP (complex).
"""
import os

import numpy as np

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "cascade.npz")


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _forward(P, x):
    return _softmax(np.maximum(0, x @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"])[0]


class CascadeRouter:
    def __init__(self, path=PATH):
        z = np.load(path)
        self.names = [str(n) for n in z["conf_names"]]
        keys = ("W1", "b1", "W2", "b2", "mu", "sd")
        self.g1 = {k: z["g1_" + k] for k in keys}
        self.g2 = {k: z["g2_" + k] for k in keys}

    def predict(self, feats, q_emb):
        x1 = (np.array([[float(feats[n]) for n in self.names]]) - self.g1["mu"]) / self.g1["sd"]
        p1 = _forward(self.g1, x1)                                   # [P(NO), P(retrieve)]
        x2 = (np.asarray(q_emb, dtype=float)[None, :] - self.g2["mu"]) / self.g2["sd"]
        p2 = _forward(self.g2, x2)                                   # [P(simple), P(complex)]
        if p1[0] >= p1[1]:
            route = "NO_RETRIEVAL"
        else:
            route = "SINGLE_STEP" if p2[0] >= p2[1] else "MULTI_STEP"
        # the route comes from the gates; probs are the product of the two gates (for display only)
        probs = {"NO_RETRIEVAL": p1[0], "SINGLE_STEP": p1[1] * p2[0], "MULTI_STEP": p1[1] * p2[1]}
        return {"route": route, "probs": {k: round(float(v), 4) for k, v in probs.items()}}