"""Step 2b: strategy B, ONE BM25 search, then the LLM answers from the retrieved paragraphs. Resumable.
Runs only on queries where A was wrong.
Output: data/answers_B.csv -> query, source_dataset, retrieved_ids, answer_B, correct_B
"""
import json
import os

from bm25 import BM25
from common import DATA, is_correct, load_rows, report, save_rows
from llm import load_ask

TOP_K = 5


def main():
    with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    by_id = {d["doc_id"]: d for d in docs}
    bm25 = BM25(docs)

    wrong = [r for r in load_rows(os.path.join(DATA, "answers_A.csv")) if r["correct_A"] == "0"]
    path = os.path.join(DATA, "answers_B.csv")
    out = load_rows(path)
    done = {r["query"] for r in out}
    todo = [r for r in wrong if r["query"] not in done]
    print(f"B: A galat tha {len(wrong)} | {len(out)} already done, {len(todo)} to go")

    if todo:
        ask = load_ask()
        for i, r in enumerate(todo, 1):
            ids = [d for d, _ in bm25.search(r["query"], k=TOP_K)]
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

    report(out, "correct_B", "A galat tha, B ne sahi kiya")


if __name__ == "__main__":
    main()
