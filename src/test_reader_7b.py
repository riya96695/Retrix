"""Does a bigger reader help? Same hybrid top-5 contexts (from answers_B_hybrid.csv), Qwen2.5-7B-Instruct instead of 3B.
Every 4th query (300 queries). Greedy decoding, same prompt. Resumable.
Output: data/answers_B_7b_sample.csv, plus relaxed accuracy 3B vs 7B on the same queries, per dataset.
Run on Kaggle with GPU T4 x2: python src/test_reader_7b.py
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DATA, load_rows, relaxed_correct, save_rows

MODEL, STEP = "Qwen/Qwen2.5-7B-Instruct", 4

with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
    by_id = {d["doc_id"]: d for d in (json.loads(line) for line in f)}
A = {r["query"]: r for r in load_rows(os.path.join(DATA, "answers_A.csv"))}
rows = load_rows(os.path.join(DATA, "answers_B_hybrid.csv"))[::STEP]
path = os.path.join(DATA, "answers_B_7b_sample.csv")
out = load_rows(path)
done = {r["query"] for r in out}
todo = [r for r in rows if r["query"] not in done]
print(f"{len(rows)} sample queries | {len(out)} already done, {len(todo)} to go | GPUs: {torch.cuda.device_count()}")

if todo:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto").eval()
    for i, r in enumerate(todo, 1):
        context = "\n\n".join(f"[{by_id[d]['title']}] {by_id[d]['text']}"
                              for d in json.loads(r["retrieved_ids"]) if d in by_id)
        prompt = ("Use the context to answer. Answer in a few words, no explanation.\n\n"
                  f"Context:\n{context}\n\nQuestion: {r['query']}")
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inp, pad_token_id=tok.eos_token_id, do_sample=False, max_new_tokens=32)
        ans = tok.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        out.append({"query": r["query"], "source_dataset": r["source_dataset"], "answer_7b": ans})
        if i % 20 == 0:
            save_rows(path, out)
            print(f"{i}/{len(todo)} | {r['query'][:50]} -> {ans[:40]}")
    save_rows(path, out)

seven = {r["query"]: r["answer_7b"] for r in out}
res = {"all": [[], []], "squad": [[], []], "hotpot": [[], []]}
for r in rows:
    gold = json.loads(A[r["query"]]["gold_answers"])
    ds = "hotpot" if "hotpot" in r["source_dataset"].lower() else "squad"
    a3, a7 = relaxed_correct(r["answer_B"], gold), relaxed_correct(seven[r["query"]], gold)
    for key in ("all", ds):
        res[key][0].append(a3); res[key][1].append(a7)
print(f"\nrelaxed accuracy on the same {len(rows)} queries (hybrid contexts)")
for key, (a3, a7) in res.items():
    print(f"{key:8s} n={len(a3):>4d} | Qwen2.5-3B {sum(a3) / len(a3):.3f} | Qwen2.5-7B {sum(a7) / len(a7):.3f}")