#pragma once

#include "verbum/tensor.h"

namespace verbum {

// C = A @ B
//   A is [M, K], B is [K, N], C is [M, N]
void matmul(const Tensor& a, const Tensor& b, Tensor& c);

// C = A @ B^T
//   A is [M, K], B is [N, K], C is [M, N]
//
// This is the one the model actually uses. HuggingFace stores linear weights
// as [out_features, in_features], so a layer is y = x @ W^T, not x @ W. Doing
// it this way means never transposing a weight matrix at load time -- and it
// happens to be the faster layout anyway, since both A and B get walked along
// their rows.
void matmul_nt(const Tensor& a, const Tensor& b, Tensor& c);

}  // namespace verbum