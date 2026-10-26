"""Does dense / hybrid retrieval find the answer more often than BM25?
Proxy metric, no gold paragraph ids needed: answer recall@k = the gold answer text appears in one of the top-k paragraphs.
Yes/no questions are skipped. Dense = frozen bge-small cosine (numpy), hybrid = reciprocal rank fusion of BM25 and dense.
Corpus embeddings are cached in data/emb_corpus.npy. Run: python src/exp_hybrid_retrieval.py
"""
import json
import os
import re
import string

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import embed as E
from bm25 import BM25
from common import DATA, load_rows

KS, POOL, RRF_K, MAXLEN = (1, 3, 5, 10), 100, 60, 256
PREFIX = "Represent this sentence for searching relevant passages: "
PUNCT = set(string.punctuation)


def norm(s):
    s = "".join(ch for ch in s.lower() if ch not in PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


@torch.no_grad()
def encode(tok, model, texts, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAXLEN,
                  return_tensors="pt").to(E.DEVICE)
        h = model(**enc).last_hidden_state[:, 0]
        out.append((h / h.norm(dim=1, keepdim=True)).cpu().numpy())
        print(f"  {min(i + bs, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.concatenate(out)


with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]
ids = [d["doc_id"] for d in docs]
pos = {d: i for i, d in enumerate(ids)}
ndoc = [norm(d["title"] + " " + d["text"]) for d in docs]
bm25 = BM25(docs)

tok = AutoTokenizer.from_pretrained(E.MODEL)
model = AutoModel.from_pretrained(E.MODEL).to(E.DEVICE).eval()
path = os.path.join(DATA, "emb_corpus.npy")
if os.path.exists(path) and len(np.load(path)) == len(docs):
    D = np.load(path)
else:
    print("embedding the corpus (first time only)")
    D = encode(tok, model, [d["title"] + ". " + d["text"] for d in docs])
    np.save(path, D)

rows = load_rows(os.path.join(DATA, "answers_A.csv"))
items = []
for r in rows:
    gold = [norm(g) for g in json.loads(r["gold_answers"])]
    gold = [g for g in gold if g and g not in ("yes", "no")]
    if gold:
        items.append((r["query"], gold, "hotpot" if "hotpot" in r["source_dataset"].lower() else "squad"))
print(f"{len(items)} queries with a non-yes/no gold answer")
Qv = encode(tok, model, [PREFIX + q for q, _, _ in items])

hit = {m: {ds: {k: [] for k in KS} for ds in ("all", "squad", "hotpot")} for m in ("BM25", "dense", "hybrid (RRF)")}
for (q, gold, ds), qv in zip(items, Qv):
    b_rank = [pos[d] for d, _ in bm25.search(q, k=POOL)]
    d_rank = list(np.argsort(-(D @ qv))[:POOL])
    fused = {}
    for ranking in (b_rank, d_rank):
        for r, i in enumerate(ranking, 1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)
    h_rank = sorted(fused, key=lambda i: -fused[i])
    for m, ranking in (("BM25", b_rank), ("dense", d_rank), ("hybrid (RRF)", h_rank)):
        for k in KS:
            ok = any(g in ndoc[i] for i in ranking[:k] for g in gold)
            hit[m]["all"][k].append(ok); hit[m][ds][k].append(ok)

print(f"\n{'method':14s}" + "".join(f"{'all@' + str(k):>9s}" for k in KS) + f"{'squad@5':>10s}{'hotpot@5':>10s}")
for m in hit:
    print(f"{m:14s}" + "".join(f"{np.mean(hit[m]['all'][k]):>9.3f}" for k in KS)
          + f"{np.mean(hit[m]['squad'][5]):>10.3f}{np.mean(hit[m]['hotpot'][5]):>10.3f}")