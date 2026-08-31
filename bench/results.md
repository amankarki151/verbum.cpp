# Benchmark results

## CPU matmul baseline (Day 2)

Naive matmul_nt, 256x1024 @ 1024x1024. Single-threaded, i-k-j loop order,
no SIMD, no blocking. This is the correctness reference, not a fast
implementation.

Machine: MacBook Pro (x86_64), AppleClang 21.0.0

| Build | Time | Throughput |
|---|---|---|
| Release (-O3) | 254.3 ms | 2.11 GFLOP/s |
| Debug (-O0, ASan + UBSan) | 1870.0 ms | 0.29 GFLOP/s |

The Release number is the real baseline — Day 7's CUDA matmul kernel gets
compared against 2.11 GFLOP/s. The Debug figure is here only as a reminder
of how much the sanitizers cost, so I don't accidentally benchmark the wrong
build later.

# Benchmark results

## CPU matmul baseline (Day 2)

Naive matmul_nt, 256x1024 @ 1024x1024. Single-threaded, i-k-j loop order,
no SIMD, no blocking. This is the correctness reference, not a fast
implementation.

Machine: MacBook Pro, Apple Silicon (arm64)

| Build | Time | Throughput |
|---|---|---|
| Release (-O3) | 254.3 ms | 2.11 GFLOP/s |
| Debug (-O0, ASan + UBSan) | 1870.0 ms | 0.29 GFLOP/s |

The Release number is the real baseline -- Day 7/8's CUDA matmul kernel gets
compared against 2.11 GFLOP/s.

## CPU generation baseline (Day 6)

Qwen3-0.6B, f32 weights (bf16 widened at load, no quantization yet),
single-threaded, KV-cache enabled, Release build, Apple Silicon (arm64).

| Prompt | Prefill | Decode |
|---|---|---|
| "The capital of France is" (5 tok) | 0.78 tok/s | 1.74 tok/s |
| "Once upon a time" (4 tok, 60 generated) | 1.16 tok/s | 1.78 tok/s |
| "def fibonacci(n):" (4 tok, 80 generated, t=0.3) | 1.26 tok/s | 1.81 tok/s |

**Decode average: ~1.78 tok/s.** This is the number Day 8's CUDA kernels get
measured against.

Known limitation, noted honestly rather than hidden: prefill is currently
*slower* than decode, which is backwards from what batched prefill should
give. The `generate` CLI's prefill loop calls the single-token cached path
(`model.step()`) once per prompt token instead of running the prompt through
the batched `model.forward()` -- so prefill isn't actually parallelized right
now. Fixing that (batch the prompt through `forward()`, then switch to
`step()` for decoding) is a real optimization, left for later rather than
blocking Day 6, which was about correctness and first real output.

### Sample output (greedy, `-p "The capital of France is" -n 20 --greedy`)

> The capital of France is Paris. The capital of Italy is Rome. The capital
> of Spain is Madrid. The capital of China

Deterministic, coherent, and factually correct on the first three -- good
confirmation that Day 5's verified forward pass is doing real work, not just
passing a logit-diff test.