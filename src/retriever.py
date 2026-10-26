"""Hybrid retriever: BM25 (scratch) + dense bge-small cosine (frozen pretrained encoder, numpy), fused with RRF.
Same interface as BM25: search(query, k) -> [(doc_id, score)], best first. The score is the RRF score.
Corpus embeddings are cached in data/emb_corpus.npy (same file that exp_hybrid_retrieval.py writes).
"""
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import embed as E
from bm25 import BM25
from common import DATA

PREFIX = "Represent this sentence for searching relevant passages: "
MAXLEN, POOL, RRF_K = 256, 100, 60


class HybridRetriever:
    def __init__(self, docs):
        self.ids = [d["doc_id"] for d in docs]
        self.bm25 = BM25(docs)
        self.tok = AutoTokenizer.from_pretrained(E.MODEL)
        self.model = AutoModel.from_pretrained(E.MODEL).to(E.DEVICE).eval()
        path = os.path.join(DATA, "emb_corpus.npy")
        if os.path.exists(path) and len(np.load(path)) == len(docs):
            self.D = np.load(path)
        else:
            self.D = self._encode([d["title"] + ". " + d["text"] for d in docs])
            np.save(path, self.D)

    @torch.no_grad()
    def _encode(self, texts, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAXLEN,
                           return_tensors="pt").to(E.DEVICE)
            h = self.model(**enc).last_hidden_state[:, 0]
            out.append((h / h.norm(dim=1, keepdim=True)).cpu().numpy())
        return np.concatenate(out)

    def search(self, query, k=5):
        bm = [d for d, _ in self.bm25.search(query, k=POOL)]
        qv = self._encode([PREFIX + query])[0]
        dense = [self.ids[i] for i in np.argsort(-(self.D @ qv))[:POOL]]
        fused = {}
        for ranking in (bm, dense):
            for r, d in enumerate(ranking, 1):
                fused[d] = fused.get(d, 0.0) + 1.0 / (RRF_K + r)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]