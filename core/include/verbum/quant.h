#pragma once

#include <cstdint>
#include <vector>

#include "verbum/tensor.h"

namespace verbum {

// A weight matrix stored as int8 with one scale per row.
//
// Per-row (not per-tensor) because output channels genuinely differ in
// magnitude -- some rows of a projection matrix have values an order of
// magnitude smaller than others. One shared scale would spend most of the
// int8 range on the loud rows and crush the quiet ones. Measured on
// synthetic weights with a deliberate 50x spread between rows: per-row
// scales give 0.80% relative RMS error, a single global scale gives 1.33%.
struct QuantTensor {
    std::vector<int8_t> data;    // [rows, cols]
    std::vector<float> scales;   // [rows]
    int64_t rows = 0;
    int64_t cols = 0;

    size_t bytes() const {
        return data.size() * sizeof(int8_t) + scales.size() * sizeof(float);
    }
};

// Symmetric quantization: no zero-point, so zero maps exactly to zero.
// Asymmetric (with a zero-point) fits skewed distributions slightly better
// but costs an extra term in every dot product. Transformer weights are
// roughly zero-centred, so symmetric is the right trade here.
QuantTensor quantize_rows(const Tensor& w);

// C = A @ Bq^T, dequantizing on the fly.
//
// The scale factors out of the inner loop: sum_k a[k]*w[j][k] equals
// scale[j] * sum_k a[k]*q[j][k]. So it's one multiply per output element,
// not one per multiply-accumulate.
void matmul_nt_q8(const Tensor& a, const QuantTensor& b, Tensor& c);

// Dequantize back to f32. For testing and for measuring round-trip error.
Tensor dequantize(const QuantTensor& q);

}  // namespace verbum