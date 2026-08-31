#pragma once

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "verbum/attention.h"
#include "verbum/tensor.h"

namespace verbum {

// Stores the key and value vectors for every token seen so far, one cache per
// layer. Without this, generating token N means recomputing attention over
// all N-1 previous tokens from scratch every single step -- O(N^3) for a
// sequence instead of O(N^2). The keys and values for old tokens never
// change, so there's no reason to recompute them.
struct KVCache {
    int n_kv_heads = 0;
    int head_dim = 0;
    int max_seq = 0;
    int len = 0;                 // how many tokens are actually stored

    std::vector<float> k;        // [max_seq, n_kv_heads * head_dim]
    std::vector<float> v;

    void init(int n_kv_heads_, int head_dim_, int max_seq_);
    void clear() { len = 0; }
};

// One token through attention, reading and appending to the cache.
// x is [1, hidden], out is [1, hidden]. The token's position is cache.len,
// which is also where its k/v get written.
void attention_step(const Tensor& x, const AttentionWeights& w,
                    const AttentionConfig& cfg, const RopeTable& rope,
                    KVCache& cache, Tensor& out);

// Sampling knobs. temperature <= 0 means greedy (always take the argmax),
// which is also what top_k == 1 gives you.
struct SamplerConfig {
    float temperature = 0.8f;
    int top_k = 40;         // 0 disables
    float top_p = 0.95f;    // 1.0 disables
    uint64_t seed = 0;
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& cfg);

    // logits is one row, [vocab]. Returns the chosen token id.
    int sample(const float* logits, int64_t vocab);

private:
    SamplerConfig cfg_;
    std::mt19937_64 rng_;
    std::vector<std::pair<float, int>> scratch_;
};

}  // namespace verbum