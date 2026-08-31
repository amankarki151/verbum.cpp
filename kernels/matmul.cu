#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define TILE 16

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA error %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while (0)

// C[M,N] = A[M,K] @ B[N,K]^T
//
// Same matmul_nt as the CPU version -- B arrives pre-transposed because
// HuggingFace stores linear weights as [out_features, in_features].
//
// One thread per output element. Every thread walks a full row of A and a
// full row of B. That means the same A row gets re-read from global memory
// by all N threads in that row, and the same B row by all M threads in that
// column. Enormously wasteful, and exactly the thing the tiled version fixes.
__global__ void matmul_nt_naive(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[col * K + k];
    }
    C[row * N + col] = sum;
}

// Same math, but each block cooperatively stages a TILE x TILE chunk of A and
// of B into shared memory first, so each element gets read from global memory
// once per block instead of once per thread.
//
// The +1 padding on the shared arrays is not a typo. Shared memory is split
// into 32 banks, 4 bytes wide. With a plain [TILE][TILE] array, threads in a
// warp reading Bs[tx][k] with consecutive tx hit addresses TILE floats apart,
// which lands them all on the same bank -- serialised access. Padding the row
// to TILE+1 makes the stride 17 instead of 16, so consecutive threads land on
// different banks.
__global__ void matmul_nt_tiled(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int ty = threadIdx.y, tx = threadIdx.x;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int nTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < nTiles; t++) {
        const int aCol = t * TILE + tx;
        const int bCol = t * TILE + tx;

        // Both loads are coalesced: consecutive tx reads consecutive k, which
        // is contiguous in memory for both A and B.
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        const int bRow = blockIdx.x * TILE + ty;
        Bs[ty][tx] = (bRow < N && bCol < K) ? B[bRow * K + bCol] : 0.0f;

        __syncthreads();

        // Zero-padding above means out-of-range elements contribute nothing,
        // so no bounds check is needed in here.
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += As[ty][k] * Bs[tx][k];
        }

        // Second barrier matters as much as the first: without it, a fast
        // thread starts overwriting the tile for iteration t+1 while a slow
        // one is still reading iteration t.
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

// ---- CPU reference, same i-k-j-free dot product form as the engine's ----
static void cpu_matmul_nt(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += A[i * K + k] * B[j * K + k];
            C[i * N + j] = s;
        }
}

struct Timing { float ms; double gflops; };

template <typename Launch>
static Timing time_kernel(Launch launch, int M, int N, int K, int iters = 20) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch();                      // warmup -- first launch pays JIT and cache costs
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    double gflop = 2.0 * M * N * K / 1e9;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return {ms, gflop / (ms / 1000.0)};
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs, %.1f GB\n\n", prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);

    // ---- correctness first, on small and deliberately ragged sizes ----
    printf("correctness vs cpu reference\n");
    int cases[][3] = {{16,16,16}, {17,13,23}, {64,64,64}, {100,70,45}, {1,1,1}};
    bool all_ok = true;

    for (auto& c : cases) {
        int M = c[0], N = c[1], K = c[2];
        std::vector<float> hA(M*K), hB(N*K), hRef(M*N), hOut(M*N);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        for (auto& x : hA) x = d(rng);
        for (auto& x : hB) x = d(rng);
        cpu_matmul_nt(hA.data(), hB.data(), hRef.data(), M, N, K);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, hA.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, hB.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, hOut.size()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size()*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size()*sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1)/TILE, (M + TILE - 1)/TILE);

        for (int which = 0; which < 2; which++) {
            CUDA_CHECK(cudaMemset(dC, 0, hOut.size()*sizeof(float)));
            if (which == 0) matmul_nt_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
            else            matmul_nt_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(hOut.data(), dC, hOut.size()*sizeof(float), cudaMemcpyDeviceToHost));

            float worst = 0;
            for (int i = 0; i < M*N; i++) worst = fmaxf(worst, fabsf(hRef[i] - hOut[i]));
            bool ok = worst < 1e-3f;
            all_ok &= ok;
            printf("  %-5s %-6s M=%3d N=%3d K=%3d  maxdiff=%.7f\n",
                   ok ? "ok" : "FAIL", which ? "tiled" : "naive", M, N, K, worst);
        }

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    }

    if (!all_ok) { printf("\ncorrectness failed -- not benchmarking a wrong kernel\n"); return 1; }

    // ---- benchmark, same shape as the Day 2 CPU baseline ----
    printf("\nbenchmark: 256x1024 @ 1024x1024 (same shape as the cpu baseline)\n");
    {
        int M = 256, K = 1024, N = 1024;
        std::vector<float> hA(M*K), hB(N*K);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        for (auto& x : hA) x = d(rng);
        for (auto& x : hB) x = d(rng);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, hA.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, hB.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, (size_t)M*N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size()*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size()*sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1)/TILE, (M + TILE - 1)/TILE);

        auto naive = time_kernel([&]{ matmul_nt_naive<<<grid, block>>>(dA, dB, dC, M, N, K); }, M, N, K);
        auto tiled = time_kernel([&]{ matmul_nt_tiled<<<grid, block>>>(dA, dB, dC, M, N, K); }, M, N, K);

        printf("  naive  %8.3f ms   %8.2f GFLOP/s\n", naive.ms, naive.gflops);
        printf("  tiled  %8.3f ms   %8.2f GFLOP/s   (%.2fx over naive)\n",
               tiled.ms, tiled.gflops, naive.ms / tiled.ms);
        printf("\n  cpu baseline was 254.3 ms / 2.11 GFLOP/s\n");
        printf("  tiled speedup over cpu: %.1fx\n", 254.3 / tiled.ms);

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    }
    return 0;
}