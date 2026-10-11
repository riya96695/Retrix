"""Step 2d: final labels from A, B, C.
Writes data/labels.csv (strict match) and data/labels_relaxed.csv (relaxed match).
Columns: query, source_dataset, hypo_answer, label, label_source
"""
import json
import os
from collections import Counter

from common import DATA, is_correct, load_rows, relaxed_correct, save_rows

PRIOR = {"squad": "SINGLE_STEP", "hotpotqa": "MULTI_STEP"}  # fallback only


def label_all(A, B, C, match):
    out = []
    for r in A:
        q, gold = r["query"], json.loads(r["gold_answers"])
        if match(r["hypo_answer"], gold):
            label, src = "NO_RETRIEVAL", "outcome"
        elif q in B and match(B[q]["answer_B"], gold):
            label, src = "SINGLE_STEP", "outcome"
        elif q in C and match(C[q]["answer_C"], gold):
            label, src = "MULTI_STEP", "outcome"
        else:
            label, src = PRIOR[r["source_dataset"]], "fallback"
        out.append({"query": q, "source_dataset": r["source_dataset"],
                    "hypo_answer": r["hypo_answer"], "label": label, "label_source": src})
    return out


def summary(name, out):
    print(f"--- {name} ---")
    print("label counts:", dict(Counter(r["label"] for r in out)))
    print("by source   :", dict(Counter((r["label"], r["label_source"]) for r in out)))
    print("fallback    :", dict(Counter(r["source_dataset"] for r in out if r["label_source"] == "fallback")))


def main():
    A = load_rows(os.path.join(DATA, "answers_A.csv"))
    B = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_B.csv"))}
    C = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_C.csv"))}

    strict = label_all(A, B, C, is_correct)
    relaxed = label_all(A, B, C, relaxed_correct)
    save_rows(os.path.join(DATA, "labels.csv"), strict)
    save_rows(os.path.join(DATA, "labels_relaxed.csv"), relaxed)

    summary("strict", strict)
    summary("relaxed", relaxed)
    print("labels changed by relaxed matcher:",
          sum(s["label"] != r["label"] for s, r in zip(strict, relaxed)))


if __name__ == "__main__":
    main()
