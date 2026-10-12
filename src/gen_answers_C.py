"""Step 2c: strategy C, iterative retrieval (the LLM writes follow-up searches, max 3 rounds). Resumable.
Runs only on queries where A and B were both wrong, or on ALL queries with --all (-> answers_C_all.csv).
Output: data/answers_C.csv -> query, source_dataset, n_rounds, answer_C, correct_C
"""
import json
import os
import sys

from bm25 import BM25
from common import DATA, is_correct, load_rows, report, save_rows
from llm import load_ask

MAX_ROUNDS = 3
TOP_K = 3
ALL = "--all" in sys.argv   # run on every query (answers_C_all.csv), not only where A and B failed


def main():
    with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    by_id = {d["doc_id"]: d for d in docs}
    bm25 = BM25(docs)

    gold = {r["query"]: r["gold_answers"] for r in load_rows(os.path.join(DATA, "answers_A.csv"))}
    if ALL:
        wrong = load_rows(os.path.join(DATA, "answers_A.csv"))
    else:
        wrong = [r for r in load_rows(os.path.join(DATA, "answers_B.csv")) if r["correct_B"] == "0"]
    path = os.path.join(DATA, "answers_C_all.csv" if ALL else "answers_C.csv")
    out = load_rows(path)
    if ALL and not out:                       # same model + settings -> same answers: reuse them
        out = load_rows(os.path.join(DATA, "answers_C.csv"))
    done = {r["query"] for r in out}
    todo = [r for r in wrong if r["query"] not in done]
    print(f"C: A aur B dono galat {len(wrong)} | {len(out)} already done, {len(todo)} to go")

    if todo:
        ask = load_ask()
        for i, r in enumerate(todo, 1):
            q = r["query"]
            search_q, seen, answer, rounds = q, [], "", 0
            for rnd in range(1, MAX_ROUNDS + 1):
                rounds = rnd
                for doc_id, _ in bm25.search(search_q, k=TOP_K):
                    if doc_id not in seen:
                        seen.append(doc_id)
                context = "\n\n".join(f"[{by_id[d]['title']}] {by_id[d]['text']}" for d in seen)
                last = rnd == MAX_ROUNDS
                if last:
                    instr = "Answer the question in a few words, no explanation."
                else:
                    instr = ("If the context is enough, reply exactly 'ANSWER: <short answer>'. "
                             "Otherwise reply exactly 'SEARCH: <short new search query for the missing fact>'.")
                reply = ask(f"{instr}\n\nContext:\n{context}\n\nQuestion: {q}")
                if not last and reply.upper().startswith("SEARCH:"):
                    search_q = reply[7:].strip() or q
                    continue
                answer = reply.split(":", 1)[1].strip() if reply.upper().startswith("ANSWER:") else reply
                break

            ok = is_correct(answer, json.loads(gold[q]))
            out.append({"query": q, "source_dataset": r["source_dataset"], "n_rounds": rounds,
                        "answer_C": answer, "correct_C": int(ok)})
            if i % 10 == 0:
                save_rows(path, out)
                print(f"{i}/{len(todo)} | rounds={rounds} | {q[:45]} -> {answer[:35]} | {ok}")
        save_rows(path, out)

    report(out, "correct_C", "A, B galat tha, C ne sahi kiya")


if __name__ == "__main__":
    main()
