#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#define BLOCK 256

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA error %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// RMSNorm. One block per row, BLOCK threads cooperating on the sum of squares.
//
// The reduction is the interesting part. Each thread first accumulates a
// strided slice of the row (thread 0 takes elements 0, 256, 512...), then a
// shared-memory tree reduction halves the active threads each round until
// thread 0 holds the total. log2(BLOCK) rounds instead of BLOCK sequential
// adds.
// ---------------------------------------------------------------------------
__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ w,
                               float* __restrict__ out,
                               int dim, float eps) {
    __shared__ float sdata[BLOCK];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* xr = x + (size_t)row * dim;
    float* orow = out + (size_t)row * dim;

    // phase 1: strided partial sums. Strided, not contiguous-chunked, so
    // consecutive threads read consecutive addresses -- coalesced.
    float partial = 0.0f;
    for (int i = tid; i < dim; i += BLOCK) {
        partial += xr[i] * xr[i];
    }
    sdata[tid] = partial;
    __syncthreads();

    // phase 2: tree reduction in shared memory
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // every thread reads the same value -- shared memory broadcast, no conflict
    const float scale = rsqrtf(sdata[0] / dim + eps);

    for (int i = tid; i < dim; i += BLOCK) {
        orow[i] = xr[i] * scale * w[i];
    }
}

// ---------------------------------------------------------------------------
// RoPE, applied in place to [heads, head_dim] for one token at `pos`.
//
// Split-half pairing, matching HuggingFace -- element i pairs with i + D/2,
// not with i+1. One thread per pair, so D/2 threads cover a head.
// ---------------------------------------------------------------------------
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

    // Read both halves before writing either. In CUDA this is per-thread
    // register state so there's no race, but the ordering still matters.
    const float a = row[i];
    const float b = row[i + half];
    row[i]        = a * c[i]        - b * s[i];
    row[i + half] = b * c[i + half] + a * s[i + half];
}

// ---------------------------------------------------------------------------
// SwiGLU's elementwise half: gate = silu(gate) * up
// ---------------------------------------------------------------------------
__global__ void silu_mul_kernel(float* __restrict__ gate,
                                const float* __restrict__ up, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    gate[i] = (g / (1.0f + __expf(-g))) * up[i];
}

// ---------------------------------------------------------------------------
// Residual add: a += b
// ---------------------------------------------------------------------------
__global__ void add_kernel(float* __restrict__ a,
                           const float* __restrict__ b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// ---------------------------------------------------------------------------
// Attention for one decode step. One block per query head.
//
// This is the decode path, not the batch path -- q is a single token, and the
// keys and values come from the KV cache. No causal mask needed: the cache
// only holds tokens at or before this position, so causality is structural.
//
// scores lives in shared memory, so max_n is bounded by shared memory size.
// ---------------------------------------------------------------------------
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

    // one thread per cached position: dot(q, k_j)
    for (int j = tid; j < n; j += blockDim.x) {
        const float* kj = kcache + (size_t)j * KVH * D + (size_t)kvh * D;
        float dot = 0.0f;
        for (int d = 0; d < D; d++) dot += qh[d] * kj[d];
        scores[j] = dot * inv_sqrt_d;
    }
    __syncthreads();

    // softmax, done serially by thread 0. n is small during decode (the
    // sequence so far), so parallelising this buys little and costs a
    // second reduction.
    if (tid == 0) {
        float m = scores[0];
        for (int j = 1; j < n; j++) m = fmaxf(m, scores[j]);
        float sum = 0.0f;
        for (int j = 0; j < n; j++) { scores[j] = __expf(scores[j] - m); sum += scores[j]; }
        const float inv = 1.0f / sum;
        for (int j = 0; j < n; j++) scores[j] *= inv;
    }
    __syncthreads();

    // weighted sum of v. One thread per output dimension, each accumulating
    // across all cached positions.
    float* o = out + (size_t)h * D;
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < n; j++) {
            acc += scores[j] * vcache[(size_t)j * KVH * D + (size_t)kvh * D + d];
        }
        o[d] = acc;
    }
}

// ===========================================================================
// CPU references
// ===========================================================================
static void cpu_rmsnorm(const float* x, const float* w, float* out,
                        int rows, int dim, float eps) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * dim;
        double s = 0;
        for (int i = 0; i < dim; i++) s += (double)xr[i] * xr[i];
        const float scale = 1.0f / sqrtf((float)(s / dim) + eps);
        for (int i = 0; i < dim; i++) out[(size_t)r * dim + i] = xr[i] * scale * w[i];
    }
}

static void cpu_rope(float* x, const float* c, const float* s,
                     int pos, int heads, int D) {
    const int half = D / 2;
    for (int h = 0; h < heads; h++) {
        float* row = x + (size_t)h * D;
        for (int i = 0; i < half; i++) {
            const float a = row[i], b = row[i + half];
            row[i]        = a * c[(size_t)pos*D + i]        - b * s[(size_t)pos*D + i];
            row[i + half] = b * c[(size_t)pos*D + i + half] + a * s[(size_t)pos*D + i + half];
        }
    }
}

static void cpu_attn_decode(const float* q, const float* kc, const float* vc,
                            float* out, int H, int KVH, int D, int n) {
    const int group = H / KVH;
    std::vector<float> sc(n);
    for (int h = 0; h < H; h++) {
        const int kvh = h / group;
        const float* qh = q + (size_t)h * D;
        for (int j = 0; j < n; j++) {
            const float* kj = kc + (size_t)j*KVH*D + (size_t)kvh*D;
            float dot = 0.f;
            for (int d = 0; d < D; d++) dot += qh[d] * kj[d];
            sc[j] = dot / sqrtf((float)D);
        }
        float m = *std::max_element(sc.begin(), sc.end());
        double sum = 0;
        for (int j = 0; j < n; j++) { sc[j] = expf(sc[j] - m); sum += sc[j]; }
        float* o = out + (size_t)h * D;
        for (int d = 0; d < D; d++) o[d] = 0.f;
        for (int j = 0; j < n; j++) {
            const float p = (float)(sc[j] / sum);
            const float* vj = vc + (size_t)j*KVH*D + (size_t)kvh*D;
            for (int d = 0; d < D; d++) o[d] += p * vj[d];
        }
    }
}

// ===========================================================================
static int failures = 0;

static void check(bool ok, const char* what, float diff) {
    printf("  %-5s %-28s maxdiff=%.8f\n", ok ? "ok" : "FAIL", what, diff);
    if (!ok) failures++;
}

static float maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    float w = 0;
    for (size_t i = 0; i < a.size(); i++) w = fmaxf(w, fabsf(a[i] - b[i]));
    return w;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n\n", prop.name, prop.multiProcessorCount);

    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    // ---- rmsnorm ----
    printf("rmsnorm\n");
    for (int dim : {64, 128, 1024, 3072, 100}) {
        const int rows = 4;
        std::vector<float> hx(rows*dim), hw(dim), href(rows*dim), hout(rows*dim);
        for (auto& v : hx) v = dist(rng);
        for (auto& v : hw) v = 1.0f + dist(rng)*0.1f;
        cpu_rmsnorm(hx.data(), hw.data(), href.data(), rows, dim, 1e-6f);

        float *dx, *dw, *dout;
        CUDA_CHECK(cudaMalloc(&dx, hx.size()*4));
        CUDA_CHECK(cudaMalloc(&dw, hw.size()*4));
        CUDA_CHECK(cudaMalloc(&dout, hout.size()*4));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size()*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dw, hw.data(), hw.size()*4, cudaMemcpyHostToDevice));

        rmsnorm_kernel<<<rows, BLOCK>>>(dx, dw, dout, dim, 1e-6f);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hout.data(), dout, hout.size()*4, cudaMemcpyDeviceToHost));

        char label[64]; snprintf(label, sizeof label, "dim=%d", dim);
        float d = maxdiff(href, hout);
        check(d < 1e-4f, label, d);
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dw)); CUDA_CHECK(cudaFree(dout));
    }

    // ---- rope ----
    printf("\nrope\n");
    for (int D : {8, 64, 128}) {
        const int heads = 4, maxpos = 64, pos = 7;
        std::vector<float> hc((size_t)maxpos*D), hs((size_t)maxpos*D);
        const float theta = 1e6f;
        for (int p = 0; p < maxpos; p++)
            for (int i = 0; i < D/2; i++) {
                const double inv = 1.0 / pow((double)theta, (double)(2*i)/D);
                const float c = (float)cos(p*inv), s = (float)sin(p*inv);
                hc[(size_t)p*D+i] = c; hc[(size_t)p*D+i+D/2] = c;
                hs[(size_t)p*D+i] = s; hs[(size_t)p*D+i+D/2] = s;
            }
        std::vector<float> hx((size_t)heads*D);
        for (auto& v : hx) v = dist(rng);
        std::vector<float> href = hx, hout((size_t)heads*D);
        cpu_rope(href.data(), hc.data(), hs.data(), pos, heads, D);

        float *dx, *dc, *ds;
        CUDA_CHECK(cudaMalloc(&dx, hx.size()*4));
        CUDA_CHECK(cudaMalloc(&dc, hc.size()*4));
        CUDA_CHECK(cudaMalloc(&ds, hs.size()*4));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size()*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dc, hc.data(), hc.size()*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ds, hs.data(), hs.size()*4, cudaMemcpyHostToDevice));

        const int pairs = heads * (D/2);
        rope_kernel<<<(pairs+255)/256, 256>>>(dx, dc, ds, pos, heads, D);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hout.data(), dx, hout.size()*4, cudaMemcpyDeviceToHost));

        char label[64]; snprintf(label, sizeof label, "head_dim=%d pos=%d", D, pos);
        float d = maxdiff(href, hout);
        check(d < 1e-5f, label, d);
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dc)); CUDA_CHECK(cudaFree(ds));
    }

    // ---- silu_mul and add ----
    printf("\nelementwise\n");
    {
        const int n = 3072 * 4;
        std::vector<float> hg(n), hu(n), href(n), hout(n);
        for (auto& v : hg) v = dist(rng);
        for (auto& v : hu) v = dist(rng);
        for (int i = 0; i < n; i++) href[i] = (hg[i]/(1.0f+expf(-hg[i]))) * hu[i];

        float *dg, *du;
        CUDA_CHECK(cudaMalloc(&dg, n*4)); CUDA_CHECK(cudaMalloc(&du, n*4));
        CUDA_CHECK(cudaMemcpy(dg, hg.data(), n*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(du, hu.data(), n*4, cudaMemcpyHostToDevice));
        silu_mul_kernel<<<(n+255)/256, 256>>>(dg, du, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hout.data(), dg, n*4, cudaMemcpyDeviceToHost));
        // __expf is a fast-math intrinsic, so the tolerance here is looser
        // than the other kernels on purpose.
        float d = maxdiff(href, hout);
        check(d < 1e-3f, "silu_mul", d);

        std::vector<float> ha(n), hb(n), aref(n);
        for (int i = 0; i < n; i++) { ha[i] = dist(rng); hb[i] = dist(rng); aref[i] = ha[i]+hb[i]; }
        float *da, *db;
        CUDA_CHECK(cudaMalloc(&da, n*4)); CUDA_CHECK(cudaMalloc(&db, n*4));
        CUDA_CHECK(cudaMemcpy(da, ha.data(), n*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, hb.data(), n*4, cudaMemcpyHostToDevice));
        add_kernel<<<(n+255)/256, 256>>>(da, db, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hout.data(), da, n*4, cudaMemcpyDeviceToHost));
        d = maxdiff(aref, hout);
        check(d < 1e-6f, "residual add", d);

        CUDA_CHECK(cudaFree(dg)); CUDA_CHECK(cudaFree(du));
        CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db));
    }

    // ---- attention decode, Qwen3-0.6B geometry ----
    printf("\nattention decode (H=16 KVH=8 D=128)\n");
    for (int n : {1, 5, 40, 200}) {
        const int H = 16, KVH = 8, D = 128;
        std::vector<float> hq((size_t)H*D), hk((size_t)n*KVH*D), hv((size_t)n*KVH*D);
        std::vector<float> href((size_t)H*D), hout((size_t)H*D);
        for (auto& v : hq) v = dist(rng);
        for (auto& v : hk) v = dist(rng);
        for (auto& v : hv) v = dist(rng);
        cpu_attn_decode(hq.data(), hk.data(), hv.data(), href.data(), H, KVH, D, n);

        float *dq, *dk, *dv, *dout;
        CUDA_CHECK(cudaMalloc(&dq, hq.size()*4));
        CUDA_CHECK(cudaMalloc(&dk, hk.size()*4));
        CUDA_CHECK(cudaMalloc(&dv, hv.size()*4));
        CUDA_CHECK(cudaMalloc(&dout, hout.size()*4));
        CUDA_CHECK(cudaMemcpy(dq, hq.data(), hq.size()*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dk, hk.data(), hk.size()*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv, hv.data(), hv.size()*4, cudaMemcpyHostToDevice));

        attn_decode_kernel<<<H, 128, n*sizeof(float)>>>(dq, dk, dv, dout, H, KVH, D, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hout.data(), dout, hout.size()*4, cudaMemcpyDeviceToHost));

        char label[64]; snprintf(label, sizeof label, "cache length n=%d", n);
        float d = maxdiff(href, hout);
        check(d < 1e-4f, label, d);

        CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk));
        CUDA_CHECK(cudaFree(dv)); CUDA_CHECK(cudaFree(dout));
    }

    printf("\n%s\n", failures == 0 ? "all kernels correct" : "SOMETHING FAILED");
    return failures == 0 ? 0 : 1;
}