"""Local LLM wrapper. Pretrained weights only; generation settings are ours."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
GEN = dict(do_sample=False, max_new_tokens=32)  # greedy: same answer every run
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def load_ask():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device=DEVICE, dtype=DTYPE).eval()

    def ask(prompt):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = tok(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**inputs, pad_token_id=tok.eos_token_id, **GEN)
        return tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return ask
