"""Dump reference token ids so the C++ tokenizer has something to be checked
against. Run it once, commit the output, and the C++ test reads it."""

import json
import sys

from transformers import AutoTokenizer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3-0.6b"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tests/reference_tokens.json"

CASES = [
    "Hello world!",
    "hello world",
    " hello",
    "I'm fine, thanks. How're you?",
    "42 apples and 1337 oranges",
    "def main():\n    print('hi')\n",
    "line one\n\nline three",
    "  leading and trailing  ",
    "CamelCaseIdentifier and snake_case_name",
    "unicode: café, naïve, 日本語, emoji ok",
    "<|im_start|>user\nwhat's up<|im_end|>\n",
    "a" * 200,
    "",
]

tok = AutoTokenizer.from_pretrained(MODEL_DIR)

out = []
for text in CASES:
    ids = tok.encode(text, add_special_tokens=False)
    out.append({"text": text, "ids": ids})
    print(f"{len(ids):4d} tokens  {text[:48]!r}")

with open(OUT, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"\nwrote {len(out)} cases to {OUT}")