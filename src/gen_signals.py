"""Step 2a: post-retrieval signals for every query (B-style: one BM25 search, top-5, LLM answers from context).
Reads  data/answers_B_all.csv, data/corpus.jsonl
Writes data/signals_B.csv (resumable)
Signals: bm25_* retrieval score | agree_* chunk agreement | b_* generation confidence | lex_*, sem_* query-chunk overlap
Run on a GPU: python src/gen_signals.py
"""
import itertools
import json
import os
import re

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

import embed as E
import llm as L
from bm25 import BM25
from common import DATA, load_rows, save_rows

TOP_K = 5
STOP = set("a an the of in on at to for and or is are was were be been by with as from that this "
           "which who whom what when where how did does do has have had it its his her their".split())


def words(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]


def main():
    with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    by_id = {d["doc_id"]: d for d in docs}
    bm25 = BM25(docs)

    rows = load_rows(os.path.join(DATA, "answers_B_all.csv"))
    queries = [r["query"] for r in rows]
    print(len(rows), "queries")
    hits = [bm25.search(q, k=TOP_K) for q in queries]

    # frozen bge-small embeddings: queries and every retrieved chunk
    etok = AutoTokenizer.from_pretrained(E.MODEL)
    emodel = AutoModel.from_pretrained(E.MODEL).to(E.DEVICE).eval()
    q_emb = E.embed(etok, emodel, queries)
    need = sorted({d for h in hits for d, _ in h})
    d_emb = dict(zip(need, E.embed(etok, emodel,
                                   [f"{by_id[d]['title']}. {by_id[d]['text']}" for d in need])))

    def retrieval_feats(q, qe, h):
        ids = [d for d, _ in h]
        sc = np.array([float(s) for _, s in h])
        embs = np.stack([d_emb[d] for d in ids])
        qw = set(words(q))
        texts = [f"{by_id[d]['title']} {by_id[d]['text']}" for d in ids]
        union = {w for t in texts for w in words(t)}
        top1 = set(words(texts[0]))
        sims = embs @ qe
        pair = [float(embs[i] @ embs[j]) for i, j in itertools.combinations(range(len(ids)), 2)]
        n = max(len(qw), 1)
        return {
            "bm25_top1": sc[0], "bm25_gap": sc[0] - sc[1], "bm25_mean": sc.mean(),
            "bm25_top1_per_tok": sc[0] / n,
            "agree_mean": float(np.mean(pair)), "agree_min": float(np.min(pair)),
            "lex_all": len(qw & union) / n, "lex_top1": len(qw & top1) / n,
            "sem_top1": float(sims[0]), "sem_max": float(sims.max()), "sem_mean": float(sims.mean()),
        }

    tok = AutoTokenizer.from_pretrained(L.MODEL)
    model = AutoModelForCausalLM.from_pretrained(L.MODEL).to(device=L.DEVICE, dtype=L.DTYPE).eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    def answer_feats(prompt):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(L.DEVICE)
        with torch.no_grad():
            out = model.generate(**inp, pad_token_id=tok.eos_token_id,
                                 return_dict_in_generate=True, output_scores=True, **L.GEN)
        seq = out.sequences[0, inp["input_ids"].shape[1]:]
        lp, ent = [], []
        for t, s in enumerate(out.scores):
            tid = int(seq[t])
            if tid in eos:
                break
            logp = torch.log_softmax(s[0].float(), dim=-1)
            lp.append(logp[tid].item())
            ent.append(-(logp.exp() * logp).sum().item())
        n_tok = len(lp)
        if not lp:
            lp, ent = [0.0], [0.0]
        ans = tok.decode(seq, skip_special_tokens=True).strip()
        return ans, {"b_mean_lp": float(np.mean(lp)), "b_min_lp": float(np.min(lp)),
                     "b_first_lp": lp[0], "b_mean_ent": float(np.mean(ent)),
                     "b_max_ent": float(np.max(ent)), "b_n_tok": n_tok}

    path = os.path.join(DATA, "signals_B.csv")
    out = load_rows(path)
    done = {r["query"] for r in out}
    todo = [i for i, q in enumerate(queries) if q not in done]
    print(f"{len(out)} already done, {len(todo)} to go")

    for n, i in enumerate(todo, 1):
        q, h = queries[i], hits[i]
        context = "\n\n".join(f"[{by_id[d]['title']}] {by_id[d]['text']}" for d, _ in h)
        ans, gf = answer_feats(
            "Use the context to answer. Answer in a few words, no explanation.\n\n"
            f"Context:\n{context}\n\nQuestion: {q}"
        )
        row = {"query": q, **retrieval_feats(q, q_emb[i], h), **gf,
               "same_answer": int(ans.strip() == rows[i]["answer_B"].strip())}
        out.append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
        if n % 20 == 0:
            save_rows(path, out)
            print(f"{n}/{len(todo)} | {q[:50]}")
    save_rows(path, out)
    same = sum(int(r["same_answer"]) for r in out)
    print(f"done. same answer as answers_B_all.csv: {same}/{len(out)}")


if __name__ == "__main__":
    main()