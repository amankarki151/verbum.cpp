#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "verbum/cuda_backend.h"

#define TILE 16

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s at %s:%d\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while (0)

// ===========================================================================
// Kernels -- these are the ones verified on a T4 in Days 7 and 8.
// ===========================================================================

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

        As[ty][tx] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
        const int bRow = blockIdx.x * TILE + ty;
        Bs[ty][tx] = (bRow < N && bCol < K) ? B[(size_t)bRow * K + bCol] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += As[ty][k] * Bs[tx][k];
        }

        __syncthreads();
    }

    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ w,
                               float* __restrict__ out,
                               int dim, float eps) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block = blockDim.x;

    const float* xr = x + (size_t)row * dim;
    float* orow = out + (size_t)row * dim;

    float partial = 0.0f;
    for (int i = tid; i < dim; i += block) {
        partial += xr[i] * xr[i];
    }
    sdata[tid] = partial;
    __syncthreads();

    for (int stride = block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    const float scale = rsqrtf(sdata[0] / dim + eps);

    for (int i = tid; i < dim; i += block) {
        orow[i] = xr[i] * scale * w[i];
    }
}

__global__ void rope_kernel(float* __restrict__ x,
                            const float* __restrict__ cos_tab,
                            const float* __restrict__ sin_tab,
                            int pos, int heads, int D) {
    const int half = D / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= heads * half) return;

    const int h = idx / half;
    const int i = idx % half;

    float* row = x + (size_t)h * D;
    const float* c = cos_tab + (size_t)pos * D;
    const float* s = sin_tab + (size_t)pos * D;

    const float a = row[i];
    const float b = row[i + half];
    row[i]        = a * c[i]        - b * s[i];
    row[i + half] = b * c[i + half] + a * s[i + half];
}

__global__ void silu_mul_kernel(float* __restrict__ gate,
                                const float* __restrict__ up, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    gate[i] = (g / (1.0f + __expf(-g))) * up[i];
}

__global__ void add_kernel(float* __restrict__ a,
                           const float* __restrict__ b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void attn_decode_kernel(const float* __restrict__ q,
                                   const float* __restrict__ kcache,
                                   const float* __restrict__ vcache,
                                   float* __restrict__ out,
                                   int H, int KVH, int D, int n) {
    extern __shared__ float scores[];

    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int group = H / KVH;
    const int kvh = h / group;

    const float* qh = q + (size_t)h * D;
    const float inv_sqrt_d = rsqrtf((float)D);

    for (int j = tid; j < n; j += blockDim.x) {
        const float* kj = kcache + (size_t)j * KVH * D + (size_t)kvh * D;
        float dot = 0.0f;
        for (int d = 0; d < D; d++) dot += qh[d] * kj[d];
        scores[j] = dot * inv_sqrt_d;
    }
    __syncthreads();

    if (tid == 0) {
        float m = scores[0];
        for (int j = 1; j < n; j++) m = fmaxf(m, scores[j]);
        float sum = 0.0f;
        for (int j = 0; j < n; j++) { scores[j] = __expf(scores[j] - m); sum += scores[j]; }
        const float inv = 1.0f / sum;
        for (int j = 0; j < n; j++) scores[j] *= inv;
    }
    __syncthreads();

    float* o = out + (size_t)h * D;
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < n; j++) {
            acc += scores[j] * vcache[(size_t)j * KVH * D + (size_t)kvh * D + d];
        }
        o[d] = acc;
    }
}

// Copies one token's k and v into the cache at position `pos`. Tiny, but
// doing it with a kernel avoids a device->host->device round trip.
__global__ void cache_append_kernel(float* __restrict__ kcache,
                                    float* __restrict__ vcache,
                                    const float* __restrict__ knew,
                                    const float* __restrict__ vnew,
                                    int pos, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const size_t off = (size_t)pos * width + i;
    kcache[off] = knew[i];
    vcache[off] = vnew[i];
}

// ===========================================================================
// Launchers -- plain C++ entry points, no CUDA types in the signatures.
// ===========================================================================

namespace verbum {

bool cuda_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
    return count > 0;
}

const char* cuda_device_name() {
    static char name[256] = {0};
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return "unknown";
    std::snprintf(name, sizeof name, "%s", prop.name);
    return name;
}

CudaBuffer cuda_alloc(size_t bytes) {
    CudaBuffer b;
    CUDA_CHECK(cudaMalloc(&b.ptr, bytes));
    b.bytes = bytes;
    return b;
}

void cuda_free(CudaBuffer& b) {
    if (b.ptr) {
        cudaFree(b.ptr);
        b.ptr = nullptr;
        b.bytes = 0;
    }
}

void cuda_upload(CudaBuffer& dst, const void* host, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst.ptr, host, bytes, cudaMemcpyHostToDevice));
}

void cuda_download(void* host, const CudaBuffer& src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(host, src.ptr, bytes, cudaMemcpyDeviceToHost));
}

void cuda_zero(CudaBuffer& b) {
    CUDA_CHECK(cudaMemset(b.ptr, 0, b.bytes));
}

void cuda_sync() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

size_t cuda_free_memory() {
    size_t freeb = 0, total = 0;
    if (cudaMemGetInfo(&freeb, &total) != cudaSuccess) return 0;
    return freeb;
}

void cuda_matmul_nt(const CudaBuffer& a, const CudaBuffer& b, CudaBuffer& c,
                    int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_nt_tiled<<<grid, block>>>(
        (const float*)a.ptr, (const float*)b.ptr, (float*)c.ptr, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_rmsnorm(const CudaBuffer& x, const CudaBuffer& w, CudaBuffer& out,
                  int rows, int dim, float eps) {
    const int block = 256;
    rmsnorm_kernel<<<rows, block, block * sizeof(float)>>>(
        (const float*)x.ptr, (const float*)w.ptr, (float*)out.ptr, dim, eps);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_rope(CudaBuffer& x, const CudaBuffer& cos_tab, const CudaBuffer& sin_tab,
               int pos, int heads, int D) {
    const int pairs = heads * (D / 2);
    rope_kernel<<<(pairs + 255) / 256, 256>>>(
        (float*)x.ptr, (const float*)cos_tab.ptr, (const float*)sin_tab.ptr,
        pos, heads, D);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_silu_mul(CudaBuffer& gate, const CudaBuffer& up, int n) {
    silu_mul_kernel<<<(n + 255) / 256, 256>>>(
        (float*)gate.ptr, (const float*)up.ptr, n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_add(CudaBuffer& a, const CudaBuffer& b, int n) {
    add_kernel<<<(n + 255) / 256, 256>>>((float*)a.ptr, (const float*)b.ptr, n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_attn_decode(const CudaBuffer& q, const CudaBuffer& kcache,
                      const CudaBuffer& vcache, CudaBuffer& out,
                      int H, int KVH, int D, int n) {
    attn_decode_kernel<<<H, 128, n * sizeof(float)>>>(
        (const float*)q.ptr, (const float*)kcache.ptr, (const float*)vcache.ptr,
        (float*)out.ptr, H, KVH, D, n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_cache_append(CudaBuffer& kcache, CudaBuffer& vcache,
                       const CudaBuffer& knew, const CudaBuffer& vnew,
                       int pos, int width) {
    cache_append_kernel<<<(width + 255) / 256, 256>>>(
        (float*)kcache.ptr, (float*)vcache.ptr,
        (const float*)knew.ptr, (const float*)vnew.ptr, pos, width);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace verbum