# verbum.cpp

An LLM inference engine written from scratch in C++ and CUDA. No PyTorch, no
llama.cpp, no third-party runtime doing the math — it loads published open
weights and runs the whole forward pass itself.

*Verbum* is Latin for "word", which is what the thing produces, one token at a
time.

## What it does

Loads a small open-weight model (Qwen3, ~0.6B) and runs it end to end:
safetensors parsing, BPE tokenizer, RMSNorm, RoPE, grouped-query attention,
SwiGLU feed-forward, KV-cache, INT8 quantization, sampling.

On top of the engine sits a small offline scene with a few NPCs you can talk
to. Their replies come from this engine. Their memory of what you said earlier
comes from Lattice, my vector database, embedded as a library.

Nothing here calls the network. No API keys, no cloud inference.

## Status

| Component | State |
|---|---|
| Safetensors loader | working |
| BPE tokenizer | working, matches HF reference on the test set |
| Forward pass (CPU) | working, logits match HF reference (max diff ~3e-5) |
| KV-cache + sampling | not started |
| CUDA kernels | not started |
| INT8 quantization | not started |
| NPC memory (Lattice) | not started |
| Demo shell | not started |

## Benchmarks

Nothing measured yet. This section gets filled in on Day 13 with real numbers
or it stays empty.

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## License

MIT