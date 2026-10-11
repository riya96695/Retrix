"""Day 1: sample queries and build a mini-corpus.

Outputs:
  data/queries.csv   -> query, gold_answers (json list), source_dataset, dataset_type, gold_doc_ids (json list)
  data/corpus.jsonl  -> doc_id, title, text

`datasets` is used ONLY to download the raw data. Everything else is ours.
"""
import csv
import hashlib
import json
import os

from datasets import load_dataset

SEED = 42
N_PER_DATASET = 600
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def make_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def load_squad(n):
    ds = load_dataset("rajpurkar/squad", split="validation").shuffle(seed=SEED).select(range(n))
    rows, docs = [], {}
    for ex in ds:
        doc_id = make_id(ex["context"])
        docs[doc_id] = {"title": ex["title"], "text": ex["context"]}
        rows.append({
            "query": ex["question"],
            "gold_answers": json.dumps(ex["answers"]["text"]),
            "source_dataset": "squad",
            "dataset_type": "single_hop",
            "gold_doc_ids": json.dumps([doc_id]),
        })
    return rows, docs


def load_hotpot(n):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").shuffle(seed=SEED).select(range(n))
    rows, docs = [], {}
    for ex in ds:
        gold_titles = set(ex["supporting_facts"]["title"])
        gold_ids = []
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            text = " ".join(s.strip() for s in sents)
            doc_id = make_id(title + text)
            docs[doc_id] = {"title": title, "text": text}
            if title in gold_titles:
                gold_ids.append(doc_id)
        rows.append({
            "query": ex["question"],
            "gold_answers": json.dumps([ex["answer"]]),
            "source_dataset": "hotpotqa",
            "dataset_type": "multi_hop",
            "gold_doc_ids": json.dumps(gold_ids),
        })
    return rows, docs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows, all_docs = [], {}
    for loader in (load_squad, load_hotpot):
        rows, docs = loader(N_PER_DATASET)
        all_rows += rows
        all_docs.update(docs)

    with open(os.path.join(OUT_DIR, "queries.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    with open(os.path.join(OUT_DIR, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for doc_id, d in all_docs.items():
            f.write(json.dumps({"doc_id": doc_id, **d}) + "\n")

    print(f"queries: {len(all_rows)} | corpus docs: {len(all_docs)}")


if __name__ == "__main__":
    main()
