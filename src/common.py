"""Shared helpers (no torch needed): paths, answer matching, resumable CSV I/O."""
import csv
import os
import string
from collections import Counter, defaultdict

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def norm_tokens(s):
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return [t for t in s.split() if t not in {"a", "an", "the"}]


def is_correct(pred, golds):
    """Strict: a gold answer appears as a contiguous token span in the prediction."""
    p = norm_tokens(pred)
    for g in golds:
        g = norm_tokens(g)
        if g and any(p[i:i + len(g)] == g for i in range(len(p) - len(g) + 1)):
            return True
    return False


def _stem(t):
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def _toks(s):
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return [_stem(t) for t in s.split() if t not in {"a", "an", "the"}]


def relaxed_correct(pred, golds, thr=0.8):
    """Relaxed: strict match OR >= thr of gold tokens appear in the prediction."""
    if is_correct(pred, golds):
        return True
    p = Counter(_toks(pred))
    for g in golds:
        gt = _toks(g)
        if gt and sum(min(c, p[t]) for t, c in Counter(gt).items()) / len(gt) >= thr:
            return True
    return False


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(path, rows):
    if not rows:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def report(rows, flag, title):
    stats = defaultdict(list)
    for r in rows:
        stats[r["source_dataset"]].append(int(r[flag]))
    for ds, v in stats.items():
        print(f"{ds}: {title} = {sum(v)}/{len(v)} ({sum(v) / len(v):.2f})")


def conf_features(lps, first_ent):
    """Confidence features from the log-probs of the generated answer tokens (EOS excluded)."""
    if not lps:
        return {"mean_lp": 0.0, "min_lp": 0.0, "first_lp": 0.0, "first_ent": first_ent, "n_tok": 0}
    return {"mean_lp": sum(lps) / len(lps), "min_lp": min(lps), "first_lp": lps[0],
            "first_ent": first_ent, "n_tok": len(lps)}
