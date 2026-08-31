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