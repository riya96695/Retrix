"""Sanity check: what fraction of gold paragraphs does BM25 find in the top-k?"""
import csv
import json
import os
from collections import defaultdict

from bm25 import BM25

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
KS = (1, 5)


def main():
    with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    with open(os.path.join(DATA, "queries.csv"), encoding="utf-8") as f:
        queries = list(csv.DictReader(f))

    bm25 = BM25(docs)
    hits = {k: defaultdict(list) for k in KS}  # k -> dataset -> [recall per query]

    for q in queries:
        gold = set(json.loads(q["gold_doc_ids"]))
        ranked = [doc_id for doc_id, _ in bm25.search(q["query"], k=max(KS))]
        for k in KS:
            found = len(gold & set(ranked[:k]))
            hits[k][q["source_dataset"]].append(found / len(gold))

    for k in KS:
        for ds, vals in hits[k].items():
            print(f"recall@{k} [{ds}]: {sum(vals) / len(vals):.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()