"""Strategy B with the hybrid retriever: ONE search (BM25 + dense, RRF), top-5, the LLM answers from the context. Resumable.
Runs on ALL queries (no reuse of the old BM25 answers). Output: data/answers_B_hybrid.csv
Columns: query, source_dataset, retrieved_ids, answer_B, correct_B. Run on a GPU: python src/gen_B_hybrid.py
"""
import json
import os

from common import DATA, is_correct, load_rows, relaxed_correct, save_rows
from llm import load_ask
from retriever import HybridRetriever

TOP_K = 5


def main():
    with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    by_id = {d["doc_id"]: d for d in docs}
    retr = HybridRetriever(docs)

    rows = load_rows(os.path.join(DATA, "answers_A.csv"))
    path = os.path.join(DATA, "answers_B_hybrid.csv")
    out = load_rows(path)
    done = {r["query"] for r in out}
    todo = [r for r in rows if r["query"] not in done]
    print(f"B hybrid: {len(rows)} queries | {len(out)} already done, {len(todo)} to go")

    if todo:
        ask = load_ask()
        for i, r in enumerate(todo, 1):
            ids = [d for d, _ in retr.search(r["query"], k=TOP_K)]
            context = "\n\n".join(f"[{by_id[d]['title']}] {by_id[d]['text']}" for d in ids)
            ans = ask(
                "Use the context to answer. Answer in a few words, no explanation.\n\n"
                f"Context:\n{context}\n\nQuestion: {r['query']}"
            )
            ok = is_correct(ans, json.loads(r["gold_answers"]))
            out.append({"query": r["query"], "source_dataset": r["source_dataset"],
                        "retrieved_ids": json.dumps(ids), "answer_B": ans, "correct_B": int(ok)})
            if i % 20 == 0:
                save_rows(path, out)
                print(f"{i}/{len(todo)} | {r['query'][:50]} -> {ans[:40]} | {ok}")
        save_rows(path, out)

    gold = {r["query"]: json.loads(r["gold_answers"]) for r in rows}
    old = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_B_all.csv"))}
    new_ok = [relaxed_correct(r["answer_B"], gold[r["query"]]) for r in out]
    old_ok = [relaxed_correct(old[r["query"]]["answer_B"], gold[r["query"]]) for r in out if r["query"] in old]
    print(f"\nrelaxed accuracy of B: BM25 {sum(old_ok) / max(len(old_ok), 1):.3f} "
          f"| hybrid {sum(new_ok) / max(len(new_ok), 1):.3f}")


if __name__ == "__main__":
    main()