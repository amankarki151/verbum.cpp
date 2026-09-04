#pragma once

#include <cstddef>

namespace verbum {

struct CudaBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
};

bool cuda_available();
const char* cuda_device_name();
size_t cuda_free_memory();

CudaBuffer cuda_alloc(size_t bytes);
void cuda_free(CudaBuffer& b);
void cuda_upload(CudaBuffer& dst, const void* host, size_t bytes);
void cuda_download(void* host, const CudaBuffer& src, size_t bytes);
void cuda_zero(CudaBuffer& b);
void cuda_sync();

void cuda_matmul_nt(const CudaBuffer& a, const CudaBuffer& b, CudaBuffer& c,
                    int M, int N, int K);

void cuda_rmsnorm(const CudaBuffer& x, const CudaBuffer& w, CudaBuffer& out,
                  int rows, int dim, float eps);

void cuda_rope(CudaBuffer& x, const CudaBuffer& cos_tab, const CudaBuffer& sin_tab,
               int pos, int heads, int D);

void cuda_silu_mul(CudaBuffer& gate, const CudaBuffer& up, int n);

void cuda_add(CudaBuffer& a, const CudaBuffer& b, int n);

void cuda_attn_decode(const CudaBuffer& q, const CudaBuffer& kcache,
                      const CudaBuffer& vcache, CudaBuffer& out,
                      int H, int KVH, int D, int n);

void cuda_cache_append(CudaBuffer& kcache, CudaBuffer& vcache,
                       const CudaBuffer& knew, const CudaBuffer& vnew,
                       int pos, int width);

}  // namespace verbum