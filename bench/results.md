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