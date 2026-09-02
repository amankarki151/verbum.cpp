#include "verbum/quant.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace verbum {

QuantTensor quantize_rows(const Tensor& w) {
    if (w.rank() != 2) throw std::runtime_error("quantize_rows wants a 2-D tensor");

    QuantTensor q;
    q.rows = w.dim(0);
    q.cols = w.dim(1);
    q.data.resize(static_cast<size_t>(q.rows) * q.cols);
    q.scales.resize(static_cast<size_t>(q.rows));

    for (int64_t r = 0; r < q.rows; r++) {
        const float* row = w.data() + r * q.cols;

        float amax = 0.0f;
        for (int64_t c = 0; c < q.cols; c++) amax = std::max(amax, std::fabs(row[c]));

        // 127, not 128 -- symmetric range is [-127, 127] so positive and
        // negative saturate at the same magnitude. Using -128 would make the
        // quantizer asymmetric by one step.
        const float scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        q.scales[static_cast<size_t>(r)] = scale;

        const float inv = 1.0f / scale;
        for (int64_t c = 0; c < q.cols; c++) {
            // lrintf rounds to nearest-even, which has no systematic bias.
            // Truncating toward zero would drag every weight slightly
            // downward -- a bias that compounds across 28 layers.
            int v = static_cast<int>(std::lrintf(row[c] * inv));
            v = std::max(-127, std::min(127, v));
            q.data[static_cast<size_t>(r) * q.cols + c] = static_cast<int8_t>(v);
        }
    }
    return q;
}

void matmul_nt_q8(const Tensor& a, const QuantTensor& b, Tensor& c) {
    if (a.rank() != 2 || c.rank() != 2) {
        throw std::runtime_error("matmul_nt_q8 wants 2-D tensors");
    }
    const int64_t M = a.dim(0), K = a.dim(1);
    const int64_t N = b.rows;
    if (b.cols != K) throw std::runtime_error("inner dimensions don't match");
    if (c.dim(0) != M || c.dim(1) != N) {
        throw std::runtime_error("output tensor is the wrong shape");
    }

    const float* A = a.data();
    const int8_t* Bq = b.data.data();
    const float* S = b.scales.data();
    float* C = c.data();

    for (int64_t i = 0; i < M; i++) {
        const float* arow = A + i * K;
        for (int64_t j = 0; j < N; j++) {
            const int8_t* brow = Bq + j * K;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                acc += arow[k] * static_cast<float>(brow[k]);
            }
            C[i * N + j] = acc * S[j];
        }
    }
}

Tensor dequantize(const QuantTensor& q) {
    Tensor out({q.rows, q.cols});
    for (int64_t r = 0; r < q.rows; r++) {
        const float s = q.scales[static_cast<size_t>(r)];
        for (int64_t c = 0; c < q.cols; c++) {
            out.data()[r * q.cols + c] =
                static_cast<float>(q.data[static_cast<size_t>(r) * q.cols + c]) * s;
        }
    }
    return out;
}

}  // namespace verbum