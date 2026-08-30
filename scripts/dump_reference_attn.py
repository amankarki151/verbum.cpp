"""Reference values for the whole attention block, generated with numpy
following the HuggingFace Qwen3 implementation. Includes QK-Norm and
grouped-query attention.

Note the KVH values: one case uses KVH=1, where every GQA mapping happens to
be identical, so it can't catch a head-mapping bug on its own. The KVH=2
cases are the ones that actually test the grouping."""

import json
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "tests/reference_attn.json"

rng = np.random.default_rng(99)


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


def attention(x, wq, wk, wv, wo, qn, kn, H, KVH, D, eps, theta):
    seq, hidden = x.shape
    q = (x @ wq.T).reshape(seq, H, D)
    k = (x @ wk.T).reshape(seq, KVH, D)
    v = (x @ wv.T).reshape(seq, KVH, D)

    q = rmsnorm(q, qn, eps)
    k = rmsnorm(k, kn, eps)

    cos, sin = rope_tables(D, seq, theta)
    c = cos[:, None, :]
    s = sin[:, None, :]
    q = (q * c + rotate_half(q) * s).astype(np.float32)
    k = (k * c + rotate_half(k) * s).astype(np.float32)

    group = H // KVH
    out = np.zeros((seq, H * D), dtype=np.float32)
    mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)

    for h in range(H):
        kvh = h // group
        qh = q[:, h, :]
        kh = k[:, kvh, :]
        vh = v[:, kvh, :]
        scores = (qh @ kh.T) / np.sqrt(D)
        scores = np.where(mask, -np.inf, scores)
        scores = scores - scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p = p / p.sum(axis=-1, keepdims=True)
        out[:, h * D:(h + 1) * D] = (p @ vh).astype(np.float32)

    return (out @ wo.T).astype(np.float32)


cases = []
for seq, hidden, H, KVH, D, theta in [
    (3, 8, 2, 1, 4, 10000.0),
    (4, 16, 4, 2, 8, 10000.0),
    (5, 32, 4, 2, 16, 1e6),
]:
    x = rng.normal(size=(seq, hidden)).astype(np.float32) * 0.5
    wq = rng.normal(size=(H * D, hidden)).astype(np.float32) * 0.1
    wk = rng.normal(size=(KVH * D, hidden)).astype(np.float32) * 0.1
    wv = rng.normal(size=(KVH * D, hidden)).astype(np.float32) * 0.1
    wo = rng.normal(size=(hidden, H * D)).astype(np.float32) * 0.1
    qn = rng.normal(loc=1.0, scale=0.05, size=(D,)).astype(np.float32)
    kn = rng.normal(loc=1.0, scale=0.05, size=(D,)).astype(np.float32)
    y = attention(x, wq, wk, wv, wo, qn, kn, H, KVH, D, 1e-6, theta)
    cases.append({
        "seq": seq, "hidden": hidden, "H": H, "KVH": KVH, "D": D,
        "eps": 1e-6, "theta": theta,
        "x": x.flatten().tolist(),
        "wq": wq.flatten().tolist(), "wk": wk.flatten().tolist(),
        "wv": wv.flatten().tolist(), "wo": wo.flatten().tolist(),
        "qn": qn.tolist(), "kn": kn.tolist(),
        "y": y.flatten().tolist(),
    })

with open(OUT, "w") as f:
    json.dump({"attention": cases}, f)

print(f"wrote {len(cases)} attention cases to {OUT}")