#pragma once

#include <cstdint>
#include <vector>

#include "verbum/nn.h"
#include "verbum/tensor.h"

namespace verbum {

// Softmax over each row of x, in place. Subtracts the row max first --
// without that, exp() overflows to inf on large scores and the whole row
// comes back as nan.
void softmax_rows(Tensor& x);

// Weights for one attention block. All the projection matrices are stored
// the way HuggingFace ships them, [out_features, in_features], so every
// multiply here goes through matmul_nt.
struct AttentionWeights {
    Tensor q_proj;   // [n_heads * head_dim, hidden]
    Tensor k_proj;   // [n_kv_heads * head_dim, hidden]
    Tensor v_proj;   // [n_kv_heads * head_dim, hidden]
    Tensor o_proj;   // [hidden, n_heads * head_dim]

    // Qwen3 normalises q and k per head, on the head_dim axis, after
    // projection and before RoPE. Length head_dim, shared across heads.
    std::vector<float> q_norm;
    std::vector<float> k_norm;
};

struct AttentionConfig {
    int n_heads = 0;
    int n_kv_heads = 0;   // fewer than n_heads means grouped-query attention
    int head_dim = 0;
    int hidden = 0;
    float rms_eps = 1e-6f;
};

// Full attention over a sequence, no KV cache yet -- that lands on Day 6.
// x is [seq, hidden], out is [seq, hidden].
void attention_forward(const Tensor& x, const AttentionWeights& w,
                       const AttentionConfig& cfg, const RopeTable& rope,
                       Tensor& out);

}  // namespace verbum