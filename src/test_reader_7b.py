"""Does a bigger reader help? Self-contained: hybrid top-5 contexts, Qwen2.5-3B vs Qwen2.5-7B-Instruct on every 4th query.
Needs no results file from earlier runs. Greedy decoding, same prompt, resumable per model.
Output: data/reader_test_3b.csv, data/reader_test_7b.csv, plus relaxed accuracy per dataset.
Run on Kaggle with GPU T4 x2: python src/test_reader_7b.py
"""
import gc
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DATA, load_rows, relaxed_correct, save_rows
from retriever import HybridRetriever

MODELS = {"3b": "Qwen/Qwen2.5-3B-Instruct", "7b": "Qwen/Qwen2.5-7B-Instruct"}
STEP, TOP_K = 4, 5

with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]
by_id = {d["doc_id"]: d for d in docs}
rows = load_rows(os.path.join(DATA, "answers_A.csv"))[::STEP]
assert rows, "data/answers_A.csv is missing or empty on this machine"
print(f"{len(rows)} sample queries | GPUs: {torch.cuda.device_count()}")

retr = HybridRetriever(docs)
contexts = {}
for r in rows:
    ids = [d for d, _ in retr.search(r["query"], k=TOP_K)]
    contexts[r["query"]] = "\n\n".join(f"[{by_id[d]['title']}] {by_id[d]['text']}" for d in ids)
del retr
gc.collect(); torch.cuda.empty_cache()


def run(tag):
    path = os.path.join(DATA, f"reader_test_{tag}.csv")
    out = load_rows(path)
    done = {r["query"] for r in out}
    todo = [r for r in rows if r["query"] not in done]
    print(f"[{tag}] {len(out)} already done, {len(todo)} to go")
    if todo:
        tok = AutoTokenizer.from_pretrained(MODELS[tag])
        model = AutoModelForCausalLM.from_pretrained(MODELS[tag], torch_dtype=torch.float16,
                                                     device_map="auto").eval()
        for i, r in enumerate(todo, 1):
            prompt = ("Use the context to answer. Answer in a few words, no explanation.\n\n"
                      f"Context:\n{contexts[r['query']]}\n\nQuestion: {r['query']}")
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True)
            inp = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inp, pad_token_id=tok.eos_token_id, do_sample=False, max_new_tokens=32)
            ans = tok.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            out.append({"query": r["query"], "answer": ans})
            if i % 20 == 0:
                save_rows(path, out)
                print(f"[{tag}] {i}/{len(todo)} | {r['query'][:50]} -> {ans[:40]}")
        save_rows(path, out)
        del model
        gc.collect(); torch.cuda.empty_cache()
    return {r["query"]: r["answer"] for r in out}


answers = {tag: run(tag) for tag in MODELS}

res = {"all": {t: [] for t in MODELS}, "squad": {t: [] for t in MODELS}, "hotpot": {t: [] for t in MODELS}}
for r in rows:
    gold = json.loads(r["gold_answers"])
    ds = "hotpot" if "hotpot" in r["source_dataset"].lower() else "squad"
    for tag in MODELS:
        ok = relaxed_correct(answers[tag][r["query"]], gold)
        res["all"][tag].append(ok); res[ds][tag].append(ok)
print(f"\nrelaxed accuracy on the same {len(rows)} queries (hybrid contexts)")
for key, v in res.items():
    print(f"{key:8s} n={len(v['3b']):>4d} | Qwen2.5-3B {sum(v['3b']) / len(v['3b']):.3f} "
          f"| Qwen2.5-7B {sum(v['7b']) / len(v['7b']):.3f}")