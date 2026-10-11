"""BM25 retriever from scratch (pure Python, no libraries)."""
import math
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"\w+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        """docs: list of dicts with keys doc_id, title, text."""
        self.k1, self.b = k1, b
        self.doc_ids = [d["doc_id"] for d in docs]
        self.postings = defaultdict(list)  # term -> [(doc_idx, tf)]
        self.doc_len = []

        for idx, d in enumerate(docs):
            tokens = tokenize(d["title"] + " " + d["text"])
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((idx, tf))

        self.N = len(docs)
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        self.idf = {
            t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }

    def search(self, query, k=5):
        """Return top-k [(doc_id, score)] for the query."""
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for idx, tf in self.postings[term]:
                norm = 1 - self.b + self.b * self.doc_len[idx] / self.avgdl
                scores[idx] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.doc_ids[i], s) for i, s in top]
