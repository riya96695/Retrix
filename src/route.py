"""Live routing: query -> LLM answer + its confidence -> router -> route.

The LLM needs a GPU (Qwen2.5-3B), so run this on Kaggle/Colab, not on a laptop CPU.
Usage:   python src/route.py "Who wrote Hamlet?"      (or no argument for an interactive loop)

The confidence features here MUST be computed exactly like in gen_conf.py
(same prompt, greedy decoding, same EOS handling), because the router was trained on those.
"""
import json
import os
import sys

import numpy as np

from common import conf_features
from train_classifier import CONF_COLS, forward

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "router.npz")


class Router:
    """Loads models/router.npz and maps confidence features -> route + probabilities."""

    def __init__(self, path=MODEL_PATH):
        z = np.load(path)
        self.P = {k: z[k] for k in ("W1", "b1", "W2", "b2")}
        self.mu, self.sd = z["mu"], z["sd"]
        self.classes = [str(c) for c in z["classes"]]

    def predict(self, feats):
        x = np.array([[float(feats[c]) for c in CONF_COLS]])
        probs = forward(self.P, (x - self.mu) / self.sd)[2][0]
        return {"route": self.classes[int(probs.argmax())],
                "probs": {c: round(float(p), 4) for c, p in zip(self.classes, probs)}}


class QwenLLM:
    """The same model and settings used for labelling: Qwen2.5-3B-Instruct, greedy decoding."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from llm import DEVICE, DTYPE, GEN, MODEL

        self.torch, self.device, self.gen_kwargs = torch, DEVICE, GEN
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL).to(device=DEVICE, dtype=DTYPE).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.eos_token_id}
        self.eos_ids.discard(None)

    def answer_with_confidence(self, query):
        torch = self.torch
        prompt = f"Answer in a few words, no explanation.\nQuestion: {query}"
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            gen = self.model.generate(**inputs, pad_token_id=self.tok.eos_token_id,
                                      return_dict_in_generate=True, output_scores=True,
                                      **self.gen_kwargs)
        seq = gen.sequences[0, inputs["input_ids"].shape[1]:].tolist()

        lps, first_ent = [], 0.0
        for step, tid in enumerate(seq[:len(gen.scores)]):
            logp = torch.log_softmax(gen.scores[step][0].float(), dim=-1)
            if step == 0:
                first_ent = float(-(logp.exp() * logp).sum())
            if tid in self.eos_ids:
                break
            lps.append(float(logp[tid]))
        answer = self.tok.decode([t for t in seq if t not in self.eos_ids],
                                 skip_special_tokens=True).strip()
        return answer, conf_features(lps, first_ent)


class Retrix:
    def __init__(self, llm=None, router=None):
        self.llm = llm or QwenLLM()
        self.router = router or Router()

    def route(self, query):
        """Returns the dict agreed in backend/README.md."""
        answer, feats = self.llm.answer_with_confidence(query)
        out = self.router.predict(feats)
        feats = {k: round(v, 5) if isinstance(v, float) else v for k, v in feats.items()}
        return {"route": out["route"], "probs": out["probs"],
                "hypothetical_answer": answer, "confidence": feats}


def main():
    system = Retrix()
    if len(sys.argv) > 1:
        print(json.dumps(system.route(" ".join(sys.argv[1:])), indent=2, ensure_ascii=False))
        return
    while True:
        q = input("query (empty to quit): ").strip()
        if not q:
            break
        print(json.dumps(system.route(q), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
