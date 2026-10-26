"""Why does B fail? Is the gold answer inside the retrieved top-5 text (reader failure) or not (retrieval failure)?
Also: how often does C fix each kind. Yes/no gold answers are skipped (a string match means nothing there).
Run: python src/diagnose_failures.py
"""
import json
import os
import re
import string
from collections import defaultdict

from common import DATA, load_rows, relaxed_correct

PUNCT = set(string.punctuation)


def norm(s):
    s = "".join(ch for ch in s.lower() if ch not in PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
    corpus = {d["doc_id"]: d for d in (json.loads(line) for line in f)}
labels = load_rows(os.path.join(DATA, "labels_relaxed.csv"))
A = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_A.csv"))}
B = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_B_all.csv"))}
C = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_C_all.csv"))}

cnt = defaultdict(lambda: defaultdict(int))
skipped = 0
for r in labels:
    q = r["query"]
    gold = json.loads(A[q]["gold_answers"])
    if all(norm(g) in ("", "yes", "no") for g in gold):
        skipped += 1
        continue
    ds = "hotpot" if "hotpot" in A[q]["source_dataset"].lower() else "squad"
    ids = json.loads(B[q]["retrieved_ids"])
    text = norm(" ".join(corpus[d]["title"] + " " + corpus[d]["text"] for d in ids if d in corpus))
    in_ctx = any(norm(g) and norm(g) in text for g in gold)
    okB = relaxed_correct(B[q]["answer_B"], gold)
    okC = relaxed_correct(C[q]["answer_C"], gold)
    kind = "B correct" if okB else ("B wrong, answer IN context (reader failure)" if in_ctx
                                    else "B wrong, answer NOT in context (retrieval failure)")
    for key in (ds, "all"):
        cnt[key][kind] += 1
        if not okB:
            cnt[key][kind + " | C fixes it"] += int(okC)

print(f"skipped {skipped} yes/no queries\n")
for key in ("all", "squad", "hotpot"):
    c, tot = cnt[key], sum(cnt[key][k] for k in ("B correct", "B wrong, answer IN context (reader failure)",
                                                   "B wrong, answer NOT in context (retrieval failure)"))
    print(f"== {key}: {tot} queries")
    for k in ("B correct", "B wrong, answer IN context (reader failure)",
              "B wrong, answer NOT in context (retrieval failure)"):
        extra = ""
        if k != "B correct" and c[k]:
            extra = f" | C fixes {c[k + ' | C fixes it']}/{c[k]} ({c[k + ' | C fixes it'] / c[k]:.0%})"
        print(f"   {k:56s}{c[k]:>5d} ({c[k] / tot:.0%}){extra}")