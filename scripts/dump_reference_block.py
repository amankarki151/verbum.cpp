"""Reference values for the SwiGLU feed-forward and a whole transformer layer,
generated with numpy following the HuggingFace Qwen3 implementation."""

import json
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "tests/reference_block.json"
rng = np.random.default_rng(7)


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


def silu(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def ffn(x, gate, up, down):
    g = x @ gate.T
    u = x @ up.T
    return ((silu(g) * u) @ down.T).astype(np.float32)


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
        sc = (q[:, h, :] @ k[:, kvh, :].T) / np.sqrt(D)
        sc = np.where(mask, -np.inf, sc)
        sc = sc - sc.max(axis=-1, keepdims=True)
        p = np.exp(sc)
        p = p / p.sum(axis=-1, keepdims=True)
        out[:, h * D:(h + 1) * D] = (p @ v[:, kvh, :]).astype(np.float32)
    return (out @ wo.T).astype(np.float32)


ffn_cases = []
for seq, hidden, f in [(2, 8, 16), (3, 16, 32)]:
    x = rng.normal(size=(seq, hidden)).astype(np.float32) * 0.5
    g = rng.normal(size=(f, hidden)).astype(np.float32) * 0.1
    u = rng.normal(size=(f, hidden)).astype(np.float32) * 0.1
    d = rng.normal(size=(hidden, f)).astype(np.float32) * 0.1
    ffn_cases.append({
        "seq": seq, "hidden": hidden, "ffn": f,
        "x": x.flatten().tolist(), "g": g.flatten().tolist(),
        "u": u.flatten().tolist(), "d": d.flatten().tolist(),
        "y": ffn(x, g, u, d).flatten().tolist(),
    })

layer_cases = []
for seq, hidden, H, KVH, D, F, theta in [
    (3, 16, 4, 2, 8, 32, 10000.0),
    (4, 32, 4, 2, 16, 64, 1e6),
]:
    eps = 1e-6
    x = rng.normal(size=(seq, hidden)).astype(np.float32) * 0.5
    inorm = rng.normal(loc=1, scale=0.05, size=(hidden,)).astype(np.float32)
    pnorm = rng.normal(loc=1, scale=0.05, size=(hidden,)).astype(np.float32)
    wq = rng.normal(size=(H * D, hidden)).astype(np.float32) * 0.1
    wk = rng.normal(size=(KVH * D, hidden)).astype(np.float32) * 0.1
    wv = rng.normal(size=(KVH * D, hidden)).astype(np.float32) * 0.1
    wo = rng.normal(size=(hidden, H * D)).astype(np.float32) * 0.1
    qn = rng.normal(loc=1, scale=0.05, size=(D,)).astype(np.float32)
    kn = rng.normal(loc=1, scale=0.05, size=(D,)).astype(np.float32)
    g = rng.normal(size=(F, hidden)).astype(np.float32) * 0.1
    u = rng.normal(size=(F, hidden)).astype(np.float32) * 0.1
    dd = rng.normal(size=(hidden, F)).astype(np.float32) * 0.1

    h1 = x + attention(rmsnorm(x, inorm, eps), wq, wk, wv, wo, qn, kn, H, KVH, D, eps, theta)
    y = h1 + ffn(rmsnorm(h1, pnorm, eps), g, u, dd)

    layer_cases.append({
        "seq": seq, "hidden": hidden, "H": H, "KVH": KVH, "D": D, "F": F,
        "eps": eps, "theta": theta,
        "x": x.flatten().tolist(), "inorm": inorm.tolist(), "pnorm": pnorm.tolist(),
        "wq": wq.flatten().tolist(), "wk": wk.flatten().tolist(),
        "wv": wv.flatten().tolist(), "wo": wo.flatten().tolist(),
        "qn": qn.tolist(), "kn": kn.tolist(),
        "g": g.flatten().tolist(), "u": u.flatten().tolist(), "d": dd.flatten().tolist(),
        "y": y.astype(np.float32).flatten().tolist(),
    })

with open(OUT, "w") as f:
    json.dump({"ffn": ffn_cases, "layer": layer_cases}, f)

print(f"wrote {len(ffn_cases)} ffn and {len(layer_cases)} layer cases to {OUT}")