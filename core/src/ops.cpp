#include "verbum/ops.h"

#include <stdexcept>

namespace verbum {

void matmul(const Tensor& a, const Tensor& b, Tensor& c) {
    if (a.rank() != 2 || b.rank() != 2 || c.rank() != 2) {
        throw std::runtime_error("matmul wants 2-D tensors");
    }
    const int64_t M = a.dim(0), K = a.dim(1);
    const int64_t K2 = b.dim(0), N = b.dim(1);
    if (K != K2) throw std::runtime_error("inner dimensions don't match");
    if (c.dim(0) != M || c.dim(1) != N) {
        throw std::runtime_error("output tensor is the wrong shape");
    }

    const float* A = a.data();
    const float* B = b.data();
    float* C = c.data();

    // i-k-j order, not the textbook i-j-k. With k in the middle, the inner
    // loop walks B along a row (stride 1) instead of down a column (stride N),
    // so it stays inside cache lines. Same number of multiplies, roughly 9x
    // faster at 256x1024x1024. Cache locality showing up in the first real
    // math in the project.
    for (int64_t i = 0; i < M; i++) {
        float* crow = C + i * N;
        for (int64_t n = 0; n < N; n++) crow[n] = 0.0f;

        for (int64_t k = 0; k < K; k++) {
            const float aik = A[i * K + k];
            if (aik == 0.0f) continue;
            const float* brow = B + k * N;
            for (int64_t j = 0; j < N; j++) {
                crow[j] += aik * brow[j];
            }
        }
    }
}

void matmul_nt(const Tensor& a, const Tensor& b, Tensor& c) {
    if (a.rank() != 2 || b.rank() != 2 || c.rank() != 2) {
        throw std::runtime_error("matmul_nt wants 2-D tensors");
    }
    const int64_t M = a.dim(0), K = a.dim(1);
    const int64_t N = b.dim(0), K2 = b.dim(1);
    if (K != K2) throw std::runtime_error("inner dimensions don't match");
    if (c.dim(0) != M || c.dim(1) != N) {
        throw std::runtime_error("output tensor is the wrong shape");
    }

    const float* A = a.data();
    const float* B = b.data();
    float* C = c.data();

    // Straight dot products. Both operands are walked along rows, which is
    // exactly the access pattern you want -- no strided reads anywhere.
    for (int64_t i = 0; i < M; i++) {
        const float* arow = A + i * K;
        for (int64_t j = 0; j < N; j++) {
            const float* brow = B + j * K;
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += arow[k] * brow[k];
            }
            C[i * N + j] = sum;
        }
    }
}

}  // namespace verbum