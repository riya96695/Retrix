"""Step 3a: sentence embeddings from a PRETRAINED model (frozen), cached to disk.

Reads  data/labels_relaxed.csv
Writes data/emb_query.npy  (input A: query only)
       data/emb_pair.npy   (input B: query [SEP] hypothetical answer, encoded together)
       data/emb_answer.npy (hypothetical answer alone, for input C: [query ; answer] concatenated)
Same row order as labels_relaxed.csv. Runs fine on a laptop CPU (small model).
"""
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common import DATA, load_rows

MODEL = "BAAI/bge-small-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def embed(tok, model, texts_a, texts_b=None, bs=32):
    out = []
    for i in range(0, len(texts_a), bs):
        a = texts_a[i:i + bs]
        b = texts_b[i:i + bs] if texts_b is not None else None
        enc = tok(a, b, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        h = model(**enc).last_hidden_state[:, 0]          # CLS pooling (as bge expects)
        h = h / h.norm(dim=1, keepdim=True)                # L2 normalise
        out.append(h.cpu().numpy())
        print(f"  {min(i + bs, len(texts_a))}/{len(texts_a)}", end="\r")
    print()
    return np.concatenate(out)


def main():
    rows = load_rows(os.path.join(DATA, "labels_relaxed.csv"))
    queries = [r["query"] for r in rows]
    answers = [r["hypo_answer"] for r in rows]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()

    for name, args in (("query", (queries, None)), ("pair", (queries, answers)),
                       ("answer", (answers, None))):
        path = os.path.join(DATA, f"emb_{name}.npy")
        if os.path.exists(path) and len(np.load(path)) == len(rows):
            print(f"{path} already there, skipping")
            continue
        print(f"embedding: {name}")
        np.save(path, embed(tok, model, *args))
    print("done")


if __name__ == "__main__":
    main()
