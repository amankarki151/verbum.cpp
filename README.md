# verbum.cpp

An LLM inference engine written from scratch in C++ and CUDA. No PyTorch, no
llama.cpp, no third-party runtime doing the math — it loads published open
weights and runs the whole forward pass itself.

*Verbum* is Latin for "word", which is what the thing produces, one token at a
time.

## Writing

- [What Actually Happens Inside a Transformer Forward Pass](https://amankarki.hashnode.dev/what-actually-happens-inside-a-transformer-forward-pass)
- [INT8 Quantization the Second Time Around](https://amankarki.hashnode.dev/int8-quantization-the-second-time-around)

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
| KV-cache + sampling | working, cached path matches full-sequence attention exactly (0.000000 diff); first real generated text |
| CUDA kernels | working end to end -- matmul, rmsnorm, rope, swiglu elementwise, residual add, and attention-decode wired into the model's forward pass. Identical output to CPU confirmed on a T4; decode at 26.83 tok/s, 14.7x over the CPU baseline |
| INT8 quantization | working, identical output to f32 at 3.99x on quantized matrices, 1.78x whole-model with current lm_head duplication (2.24x once that's fixed) |
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