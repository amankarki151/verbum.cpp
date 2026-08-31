"""Runs the real model through HuggingFace and dumps logits. This is the
actual correctness gate -- unlike the numpy references, this is genuinely
independent of my implementation.

Only the top-k logits per position get saved; the full [seq, 151936] array is
enormous and comparing the top of the distribution is what matters anyway."""

import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3-0.6b"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tests/reference_logits.json"

PROMPTS = [
    "Hello",
    "The capital of France is",
    "def add(a, b):",
]
TOPK = 20

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
model.eval()

cases = []
for text in PROMPTS:
    ids = tok.encode(text, add_special_tokens=False)
    with torch.no_grad():
        out = model(torch.tensor([ids]))
    logits = out.logits[0].float().numpy()   # [seq, vocab]

    last = logits[-1]
    top_idx = np.argsort(-last)[:TOPK]

    cases.append({
        "text": text,
        "ids": ids,
        "top_ids": top_idx.tolist(),
        "top_logits": last[top_idx].tolist(),
        "argmax": int(np.argmax(last)),
        "argmax_token": tok.decode([int(np.argmax(last))]),
        "first_pos_sample": logits[0][:8].tolist(),
    })
    print(f"{text!r} -> next token {cases[-1]['argmax_token']!r}")

with open(OUT, "w") as f:
    json.dump(cases, f)

print(f"\nwrote {len(cases)} cases to {OUT}")