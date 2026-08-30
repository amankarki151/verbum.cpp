"""Reference values for rmsnorm and rope, generated with numpy following the
HuggingFace implementations. The C++ test checks against these.

This is a cross-implementation check, not a fully independent oracle -- the
real gate is Day 5, where the whole forward pass gets compared against actual
HuggingFace output. This catches C++-specific mistakes early so Day 5 isn't
debugging five things at once."""

import json
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "tests/reference_nn.json"

rng = np.random.default_rng(1234)


def rmsnorm(x, w, eps):
    var = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps) * w).astype(np.float32)


def rope_tables(head_dim, max_pos, theta):
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    pos = np.arange(max_pos, dtype=np.float64)[:, None]
    freqs = pos * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rope(x, pos, cos, sin):
    return (x * cos[pos] + rotate_half(x) * sin[pos]).astype(np.float32)


out = {}

rms_cases = []
for rows, dim in [(1, 8), (3, 16), (2, 128)]:
    x = rng.normal(size=(rows, dim)).astype(np.float32)
    w = rng.normal(loc=1.0, scale=0.1, size=(dim,)).astype(np.float32)
    y = rmsnorm(x, w, 1e-6)
    rms_cases.append({
        "rows": rows, "dim": dim, "eps": 1e-6,
        "x": x.flatten().tolist(),
        "w": w.tolist(),
        "y": y.flatten().tolist(),
    })
out["rmsnorm"] = rms_cases

# head_dim 128 and theta 1e6 match Qwen3-0.6B's actual config
rope_cases = []
for heads, head_dim, theta, pos in [(2, 8, 10000.0, 0), (2, 8, 10000.0, 3), (4, 128, 1e6, 7)]:
    cos, sin = rope_tables(head_dim, 64, theta)
    x = rng.normal(size=(heads, head_dim)).astype(np.float32)
    y = apply_rope(x, pos, cos, sin)
    rope_cases.append({
        "heads": heads, "head_dim": head_dim, "theta": theta, "pos": pos,
        "x": x.flatten().tolist(),
        "y": y.flatten().tolist(),
    })
out["rope"] = rope_cases

with open(OUT, "w") as f:
    json.dump(out, f)

print(f"wrote {len(rms_cases)} rmsnorm and {len(rope_cases)} rope cases to {OUT}")