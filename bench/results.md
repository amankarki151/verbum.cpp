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


## CUDA matmul (Day 7)

256x1024 @ 1024x1024, float32, run on Kaggle, Tesla T4 (40 SMs, 15.6 GB).

| Implementation | Time | Throughput | vs CPU baseline |
|---|---|---|---|
| CPU naive (Day 2, Apple Silicon) | 254.3 ms | 2.11 GFLOP/s | 1x |
| CUDA naive (TILE=16 launch config) | 11.076 ms | 48.47 GFLOP/s | 4.6x |
| CUDA tiled (TILE=16) | 0.742 ms | 723.83 GFLOP/s | 342.9x |

Tile size sweep, tiled kernel (naive kernel's launch config also uses TILE,
so its number shifts too -- not just the tiled kernel):

| TILE | naive | tiled |
|---|---|---|
| 8 | 86.49 GFLOP/s | 263.04 GFLOP/s |
| 16 | 48.47 GFLOP/s | 723.83 GFLOP/s |
| 32 | 34.20 GFLOP/s | 751.21 GFLOP/s |

Tiled throughput climbs with tile size (more shared-memory reuse per global
load); naive throughput falls (larger blocks leave fewer blocks resident per
SM, which hurts a memory-bound kernel's ability to hide latency). TILE=16 was
kept as the shipped default -- TILE=32 is marginally faster but only by about
4%, not worth the larger, less flexible block size for that little gain.

Resource usage (`--ptxas-options=-v`, TILE=16):

| Kernel | Registers/thread | Shared mem/block | Spills |
|---|---|---|---|
| tiled | 37 | 2176 bytes | 0 |
| naive | 46 | 0 (388 bytes constant mem) | 0 |

Zero spills on both confirms neither kernel is register-pressured enough to
spill to local memory, which would have quietly tanked performance without
showing up as a correctness failure.

Both kernels verified against the CPU reference on five sizes, including
non-multiples of the tile size, before any timing was taken -- max diff
1.4e-6, consistent with float accumulation order rather than a bug.

Peak FP32 throughput on a T4 is roughly 8.1 TFLOP/s. The tiled kernel's
723.83 GFLOP/s is about 9% of that ceiling -- expected for a first
hand-written shared-memory kernel with no register blocking or vectorized
loads yet, and a real number rather than something that looks suspiciously
close to peak.

Caveat worth stating plainly: the "vs CPU baseline" column compares against
my own naive single-threaded CPU implementation, not a tuned one. A blocked,
SIMD, multithreaded CPU matmul would land in the tens of GFLOP/s and close
most of that gap. The honest claim is "GPU vs my reference implementation,"
not "GPU vs what a CPU can do."

## INT8 quantization (Day 9)

Per-row symmetric INT8 quantization applied to all 28 layers' attention and
FFN projection matrices (196 tensors total: q/k/v/o_proj + gate/up/down_proj
per layer). Measured against the real Qwen3-0.6B weights, not synthetic data.

| | f32 | int8 | ratio |
|---|---|---|---|
| Quantized matrices only | 1761.61 MB | 441.78 MB | 3.99x |

Worst single tensor: `model.layers.19.mlp.down_proj.weight` at 1.20% relative
RMS error. Layer 0's range: 0.83% (k_proj) to 1.06% (o_proj). All 196
quantized tensors landed under 1.3% -- closely matching the synthetic weight
tests from the morning push (0.79-0.85%), which means real model weights
quantize just as cleanly as the Gaussian approximation predicted.

**Honest caveat on the headline number.** 3.99x is the ratio for the
quantized matrices alone. The embedding table and lm_head stay f32 -- and
embed_tokens turns out to be the single largest tensor in the entire model
(622 MB on its own), so it dominates what "whole model" actually means.

Worse: the current loader loads `lm_head.weight` as a full separate copy of
`embed_tokens.weight` even though the config says `tied_embeddings=yes` and
they're numerically identical -- an existing inefficiency from Day 5, not
something today introduced, costing an extra ~622 MB for nothing. Flagged
for a future fix, not corrected today, since fixing it isn't quantization's
job and conflating the two would muddy both numbers.

| | Total f32 (current code) | Total after quantizing | Whole-model ratio |
|---|---|---|---|
| As currently loaded (with the duplicate) | 3006.5 MB | 1686.7 MB | 1.78x |
| If the lm_head duplication were also fixed | 2384.2 MB | 1064.3 MB | 2.24x |

The honest number to lead with anywhere public is **2.24x whole-model**, not
3.99x -- that's what quantization alone actually buys once the unrelated
lm_head bug isn't inflating the "before" number.

Honest note on speed: decode throughput barely moved (1.83 -> 1.88 tok/s).
Expected -- matmul_nt_q8 converts each int8 weight to float before
multiplying, with no SIMD speedup applied. Today's result is a memory win,
not a speed win; a faster quantized matmul (vectorized int8 dot products) is
future work, not something claimed here.

## CUDA end-to-end generation (Day 10)

Qwen3-0.6B, f32 weights, Tesla T4, greedy sampling, identical output to CPU
confirmed on "The capital of France is" -> "Paris. ... Rome. ... Madrid. ...
China" -- exact match, word for word.

| | Prefill | Decode |
|---|---|---|
| CPU (this Kaggle instance) | 0.73 tok/s | 0.74 tok/s |
| CUDA (T4) | 9.66 tok/s | 26.83 tok/s |
| Speedup, same machine | 13.2x | 36.2x |

For context against the documented Mac baseline (bench/results.md, Day 6):
CPU decode there was 1.83 tok/s, so CUDA's 26.83 tok/s is about **14.7x**
over the machine this project was actually developed on -- the more
honest number to lead with publicly, since "36x" is partly this Kaggle
instance's CPU being weaker than usual, not purely the GPU's doing.

Sanity check: T4 memory bandwidth is ~320 GB/s. Decode is memory-bound --
every token reads the full ~3GB of f32 weights once -- putting the
theoretical ceiling around 106 tok/s. 26.83 tok/s is roughly 25% of that,
which is the right range for a first integration with per-op kernel
launches, no fusion, and a matmul kernel not specifically tuned for the
skinny matrix-vector shape decode uses. Real room to improve, not a red
flag on the number itself.

Prefill's smaller speedup (13.2x vs decode's 36.2x) traces back to the same
unbatched-prefill limitation noted in Day 6 -- fixed per-call overhead
matters more across 5 tokens than 20.

## Benchmark summary (Day 13, single session per machine)

50 generated tokens, greedy sampling, same prompt ("The capital of France
is"), each machine's three (or two) modes measured back to back in one
sitting -- not stitched together from different days.

| Mode | Machine | tok/s | p50 latency | p99 latency | Weights |
|---|---|---|---|---|---|
| f32 | Mac (arm64) | 1.28 | 593.0 ms | 1857.9 ms | 3006.5 MB |
| int8 | Mac (arm64) | 1.72 | 547.8 ms | 1090.3 ms | 1686.7 MB |
| f32 | Kaggle (CPU) | 0.73 | 1372.2 ms | 1401.4 ms | 3006.5 MB |
| int8 | Kaggle (CPU) | 1.05 | 948.2 ms | 1036.0 ms | 1686.7 MB |
| cuda (f32) | Kaggle T4 | 28.01 | 35.6 ms | 37.8 ms | 3006.5 MB |

**Read the CPU rows as "roughly this," not exact.** Kaggle's CPU allocation
is shared and has shown 2x+ swings across sessions earlier in this project
(0.74 to 1.83 tok/s for the same f32 workload on different days). The GPU
number is far more trustworthy as reproducible: this run's 28.01 tok/s sits
close to an earlier, completely separate session's 26.83 tok/s, and this
run's own p50/p99 are nearly identical to each other -- almost no tail
latency, unlike the CPU rows' more variable single-token timings.

**CUDA speedup, same session, apples to apples:** 38.4x over this session's
CPU f32, 26.7x over this session's CPU int8. Sanity-checked against
hardware: a T4's ~320 GB/s memory bandwidth puts a theoretical ceiling
around 106 tok/s for this memory-bound decode workload (every token reads
the full ~3GB of weights once). 28.01 tok/s is about 26% of that ceiling --
consistent with the earlier CUDA integration day's estimate, and expected
for a first integration with per-operation kernel launches and no kernel
fusion yet.

**Quantization's speed effect is larger here than Day 9's original
measurement** (Mac: 34% faster; Kaggle: 44% faster; Day 9 reported ~3%).
Plausible explanation, not a settled finding: `matmul_nt_q8` still converts
int8 to float before multiplying -- no vectorized int8 arithmetic -- but it
moves 4x less data through memory to do it, and decode is a
memory-bandwidth-bound workload. Reduced memory traffic could genuinely
speed up a bandwidth-bound matmul even with identical per-element math.
Every number here is a single run, not an averaged series, so this is an
honest observation with a plausible mechanism, not a claimed multiplier.

Earlier per-day entries below are kept for their historical detail (what
was being verified on each specific day) -- this table is the one honest,
single-session comparison to actually cite.