"""Parity check: live answer_confidence vs the offline score for the same queries. Run on a GPU."""
import csv
import json
import os

from bm25 import BM25
from common import DATA
import train_confidence as T
from live_signals import LiveConfidence
from rag import DocStore
from route import QwenLLM

with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]
store = DocStore()
store.chunks, store.index = docs, BM25(docs)

with open(os.path.join(DATA, "signals_B.csv"), encoding="utf-8") as f:
    rows = list(csv.DictReader(f))[:8]

live, model = LiveConfidence(QwenLLM()), T.load()
print(f"{'live':>8} {'offline':>8}")
for r in rows:
    _, s_live = live.answer_with_score(r["query"], store.search(r["query"], 5))
    s_off = T.score({k: float(r[k]) for k in T.FEATURES}, model)
    print(f"{s_live:>8.3f} {s_off:>8.3f} | {r['query'][:50]}")