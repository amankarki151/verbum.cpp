#pragma once

#include <cstdint>
#include <vector>

#include "verbum/tensor.h"

namespace verbum {

// out = (x / sqrt(mean(x^2) + eps)) * weight, applied per row.
// x and out are [rows, dim]; weight is [dim].
//
// No mean subtraction, unlike LayerNorm -- that's the whole "RMS" part. One
// less pass over the data and one less thing to store, and it turns out
// transformers don't miss the centering.
void rmsnorm(const Tensor& x, const std::vector<float>& weight, float eps,
             Tensor& out);

// Precomputed cos/sin tables for rotary embeddings.
struct RopeTable {
    int head_dim = 0;
    int max_pos = 0;
    std::vector<float> cos;  // [max_pos, head_dim]
    std::vector<float> sin;  // [max_pos, head_dim]
};

RopeTable build_rope_table(int head_dim, int max_pos, float theta);

// Applies RoPE in place to x, which is [num_heads, head_dim], for one token
// sitting at position `pos`.
//
// Uses the split-half pairing (i paired with i + head_dim/2), which is what
// HuggingFace does -- NOT the adjacent-pair (i, i+1) form in the original
// paper. Both are valid rotations, but the weights were trained against the
// HF one, so anything else produces plausible-looking garbage.
void apply_rope(Tensor& x, int pos, const RopeTable& table);

// Copies the embedding rows for `ids` out of an embedding matrix stored as
// [vocab, dim] floats. out is [ids.size(), dim].
void embedding_lookup(const std::vector<float>& embed_matrix, int64_t dim,
                      const std::vector<int>& ids, Tensor& out);

}  // namespace verbum