"""Everything for the 7B + hybrid pipeline in ONE pass over all 1200 queries. Resumable. Run on Kaggle GPU T4 x2.
Per query: A (no search) answer + confidence features | B (hybrid top-5) answer + 17 Step 2 signals | C (multi-step, as in rag.py).
Output: /kaggle/working/run7b.csv (on Kaggle) or data/run7b.csv. Run: python src/gen7b_all.py
"""
import itertools
import json
import os
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import embed as E
from common import DATA, conf_features, is_correct, load_rows, save_rows
from retriever import HybridRetriever

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW, TOP_K_B, TOP_K_C, MAX_ROUNDS = 32, 5, 3, 3
OUT = "/kaggle/working/run7b.csv" if os.path.isdir("/kaggle/working") else os.path.join(DATA, "run7b.csv")
STOP = set("a an the of in on at to for and or is are was were be been by with as from that this "
           "which who whom what when where how did does do has have had it its his her their".split())


def words(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]


with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]
by_id = {d["doc_id"]: d for d in docs}
retr = HybridRetriever(docs)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto").eval()
_eos = model.generation_config.eos_token_id
EOS = set(_eos if isinstance(_eos, (list, tuple)) else [_eos]) | {tok.eos_token_id}
EOS.discard(None)


@torch.no_grad()
def embed_quiet(texts, bs=32):          # same as embed.embed (bge-small, 128 tokens, CLS, L2), without printing
    out = []
    for i in range(0, len(texts), bs):
        enc = retr.tok(texts[i:i + bs], padding=True, truncation=True, max_length=128,
                       return_tensors="pt").to(E.DEVICE)
        h = retr.model(**enc).last_hidden_state[:, 0]
        out.append((h / h.norm(dim=1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


def generate(prompt, scores=False):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, pad_token_id=tok.eos_token_id, do_sample=False, max_new_tokens=MAX_NEW,
                             return_dict_in_generate=scores, output_scores=scores)
    seq = (out.sequences if scores else out)[0, inp["input_ids"].shape[1]:].tolist()
    return (seq, out.scores) if scores else tok.decode(seq, skip_special_tokens=True).strip()


def answer_conf(query):                  # same as route.py answer_with_confidence
    seq, sc = generate(f"Answer in a few words, no explanation.\nQuestion: {query}", scores=True)
    lps, first_ent = [], 0.0
    for step, tid in enumerate(seq[:len(sc)]):
        logp = torch.log_softmax(sc[step][0].float(), dim=-1)
        if step == 0:
            first_ent = float(-(logp.exp() * logp).sum())
        if tid in EOS:
            break
        lps.append(float(logp[tid]))
    ans = tok.decode([t for t in seq if t not in EOS], skip_special_tokens=True).strip()
    return ans, conf_features(lps, first_ent)


def answer_signals(query, ctx_docs):     # B answer + the 17 Step 2 signals
    context = "\n\n".join(f"[{d['title']}] {d['text']}" for d in ctx_docs)
    seq, sc = generate("Use the context to answer. Answer in a few words, no explanation.\n\n"
                       f"Context:\n{context}\n\nQuestion: {query}", scores=True)
    lp, ent = [], []
    for t, s in enumerate(sc):
        tid = seq[t]
        if tid in EOS:
            break
        logp = torch.log_softmax(s[0].float(), dim=-1)
        lp.append(logp[tid].item())
        ent.append(-(logp.exp() * logp).sum().item())
    n_tok = len(lp)
    if not lp:
        lp, ent = [0.0], [0.0]
    ans = tok.decode([t for t in seq if t not in EOS], skip_special_tokens=True).strip()
    embs = embed_quiet([f"{d['title']}. {d['text']}" for d in ctx_docs])
    qe = embed_quiet([query])[0]
    bm = np.array([float(s) for _, s in retr.bm25.search(query, k=5)])
    qw = set(words(query))
    texts = [f"{d['title']} {d['text']}" for d in ctx_docs]
    union = {w for t in texts for w in words(t)}
    sims = embs @ qe
    pair = [float(embs[i] @ embs[j]) for i, j in itertools.combinations(range(len(ctx_docs)), 2)]
    n = max(len(qw), 1)
    return ans, {
        "bm25_top1": bm[0], "bm25_gap": bm[0] - bm[1], "bm25_mean": bm.mean(), "bm25_top1_per_tok": bm[0] / n,
        "agree_mean": float(np.mean(pair)), "agree_min": float(np.min(pair)),
        "b_mean_lp": float(np.mean(lp)), "b_min_lp": float(np.min(lp)), "b_first_lp": lp[0],
        "b_mean_ent": float(np.mean(ent)), "b_max_ent": float(np.max(ent)), "b_n_tok": n_tok,
        "lex_all": len(qw & union) / n, "lex_top1": len(qw & set(words(texts[0]))) / n,
        "sem_top1": float(sims[0]), "sem_max": float(sims.max()), "sem_mean": float(sims.mean())}


def run_c(query):                        # multi-step, same prompts and limits as rag.py
    seen, search_q, final = {}, query, ""
    for rnd in range(1, MAX_ROUNDS + 1):
        for d, _ in retr.search(search_q, k=TOP_K_C):
            seen.setdefault(d, by_id[d])
        last = rnd == MAX_ROUNDS
        instr = ("Answer the question in a few words, no explanation." if last else
                 "If the context is enough, reply exactly 'ANSWER: <short answer>'. "
                 "Otherwise reply exactly 'SEARCH: <short new search query for the missing fact>'.")
        ctx = "\n\n".join(f"[{c['title']}] {c['text']}" for c in seen.values())
        reply = generate(f"{instr}\n\nContext:\n{ctx}\n\nQuestion: {query}")
        if not last and reply.upper().startswith("SEARCH:"):
            search_q = reply[7:].strip() or query
            continue
        final = reply.split(":", 1)[1].strip() if reply.upper().startswith("ANSWER:") else reply
        break
    return final, rnd


rows = load_rows(os.path.join(DATA, "answers_A.csv"))
out = load_rows(OUT)
done = {r["query"] for r in out}
todo = [r for r in rows if r["query"] not in done]
print(f"{len(rows)} queries | {len(out)} already done, {len(todo)} to go | GPUs: {torch.cuda.device_count()}")

for n, r in enumerate(todo, 1):
    q, gold = r["query"], json.loads(r["gold_answers"])
    ans_a, cf = answer_conf(q)
    ids = [d for d, _ in retr.search(q, k=TOP_K_B)]
    ans_b, sg = answer_signals(q, [by_id[d] for d in ids])
    ans_c, n_rounds = run_c(q)
    row = {"query": q, "source_dataset": r["source_dataset"], "gold_answers": r["gold_answers"],
           "ans_A": ans_a, "correct_A": int(is_correct(ans_a, gold)), **cf,
           "ans_B": ans_b, "retrieved_ids": json.dumps(ids), "correct_B": int(is_correct(ans_b, gold)), **sg,
           "ans_C": ans_c, "n_rounds": n_rounds, "correct_C": int(is_correct(ans_c, gold))}
    out.append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
    if n % 20 == 0:
        save_rows(OUT, out)
        c = lambda k: sum(int(x[k]) for x in out) / len(out)
        print(f"{n}/{len(todo)} | strict acc so far A {c('correct_A'):.3f} B {c('correct_B'):.3f} "
              f"C {c('correct_C'):.3f} | {q[:40]}")
save_rows(OUT, out)
c = lambda k: sum(int(x[k]) for x in out) / len(out)
print(f"\ndone: {len(out)} queries | strict accuracy A {c('correct_A'):.3f} B {c('correct_B'):.3f} "
      f"C {c('correct_C'):.3f} | avg C rounds {sum(int(x['n_rounds']) for x in out) / len(out):.2f}")