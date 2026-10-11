"""Step 2a: strategy A, LLM answers WITHOUT retrieval. Resumable.
Output: data/answers_A.csv -> query, gold_answers, source_dataset, hypo_answer, correct_A
"""
import json
import os

from common import DATA, is_correct, load_rows, report, save_rows
from llm import load_ask


def main():
    queries = load_rows(os.path.join(DATA, "queries.csv"))
    path = os.path.join(DATA, "answers_A.csv")
    out = load_rows(path)
    done = {r["query"] for r in out}
    todo = [r for r in queries if r["query"] not in done]
    print(f"A: {len(out)} already done, {len(todo)} to go")

    if todo:
        ask = load_ask()
        for i, r in enumerate(todo, 1):
            ans = ask(f"Answer in a few words, no explanation.\nQuestion: {r['query']}")
            ok = is_correct(ans, json.loads(r["gold_answers"]))
            out.append({"query": r["query"], "gold_answers": r["gold_answers"],
                        "source_dataset": r["source_dataset"], "hypo_answer": ans,
                        "correct_A": int(ok)})
            if i % 20 == 0:
                save_rows(path, out)
                print(f"{i}/{len(todo)} | {r['query'][:50]} -> {ans[:40]} | {ok}")
        save_rows(path, out)

    report(out, "correct_A", "LLM bina search ke sahi")


if __name__ == "__main__":
    main()
