"""Live Step 2 score for the SINGLE_STEP path.
The signals here MUST be computed exactly like gen_signals.py (same prompt, top-5 BM25, greedy decoding,
same bge-small embeddings), because models/confidence.npz was trained on those.
"""
import itertools
import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import embed as E
import train_confidence as T

STOP = set("a an the of in on at to for and or is are was were be been by with as from that this "
           "which who whom what when where how did does do has have had it its his her their".split())


def words(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]


class LiveConfidence:
    def __init__(self, llm):
        self.llm = llm                                   # QwenLLM from route.py
        self.etok = AutoTokenizer.from_pretrained(E.MODEL)
        self.emodel = AutoModel.from_pretrained(E.MODEL).to(E.DEVICE).eval()
        self.model = T.load()                            # models/confidence.npz

    def _retrieval_feats(self, query, hits):
        embs = E.embed(self.etok, self.emodel, [f"{c['title']}. {c['text']}" for c, _ in hits])
        qe = E.embed(self.etok, self.emodel, [query])[0]
        sc = np.array([float(s) for _, s in hits])
        qw = set(words(query))
        texts = [f"{c['title']} {c['text']}" for c, _ in hits]
        union = {w for t in texts for w in words(t)}
        top1 = set(words(texts[0]))
        sims = embs @ qe
        pair = [float(embs[i] @ embs[j]) for i, j in itertools.combinations(range(len(hits)), 2)]
        n = max(len(qw), 1)
        return {
            "bm25_top1": sc[0], "bm25_gap": sc[0] - sc[1], "bm25_mean": sc.mean(),
            "bm25_top1_per_tok": sc[0] / n,
            "agree_mean": float(np.mean(pair)), "agree_min": float(np.min(pair)),
            "lex_all": len(qw & union) / n, "lex_top1": len(qw & top1) / n,
            "sem_top1": float(sims[0]), "sem_max": float(sims.max()), "sem_mean": float(sims.mean()),
        }

    def answer_with_score(self, query, hits):
        """hits = [(chunk, bm25_score)] from DocStore.search. Returns (answer, answer_confidence or None)."""
        llm, tok = self.llm, self.llm.tok
        context = "\n\n".join(f"[{c['title']}] {c['text']}" for c, _ in hits)
        prompt = ("Use the context to answer. Answer in a few words, no explanation.\n\n"
                  f"Context:\n{context}\n\nQuestion: {query}")
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            out = llm.model.generate(**inp, pad_token_id=tok.eos_token_id,
                                     return_dict_in_generate=True, output_scores=True, **llm.gen_kwargs)
        seq = out.sequences[0, inp["input_ids"].shape[1]:]
        lp, ent = [], []
        for t, s in enumerate(out.scores):
            tid = int(seq[t])
            if tid in llm.eos_ids:
                break
            logp = torch.log_softmax(s[0].float(), dim=-1)
            lp.append(logp[tid].item())
            ent.append(-(logp.exp() * logp).sum().item())
        answer = tok.decode(seq, skip_special_tokens=True).strip()
        if len(hits) < 2:                                # chunk-gap / agreement need at least 2 chunks
            return answer, None
        n_tok = len(lp)
        if not lp:
            lp, ent = [0.0], [0.0]
        feats = self._retrieval_feats(query, hits)
        feats.update({"b_mean_lp": float(np.mean(lp)), "b_min_lp": float(np.min(lp)),
                      "b_first_lp": lp[0], "b_mean_ent": float(np.mean(ent)),
                      "b_max_ent": float(np.max(ent)), "b_n_tok": n_tok})
        return answer, round(T.score(feats, self.model), 4)