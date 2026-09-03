#pragma once

#include <string>
#include <vector>

#include "verbum/attention.h"
#include "verbum/config.h"
#include "verbum/generate.h"
#include "verbum/nn.h"
#include "verbum/quant.h"
#include "verbum/safetensors.h"
#include "verbum/tensor.h"

namespace verbum {

// SwiGLU feed-forward. Two parallel projections up, one back down:
//   down( silu(gate(x)) * up(x) )
// The gate branch acts as a learned per-channel valve on the up branch --
// that elementwise multiply is the "gated" part.
struct FFNWeights {
    Tensor gate_proj;   // [ffn, hidden]
    Tensor up_proj;     // [ffn, hidden]
    Tensor down_proj;   // [hidden, ffn]
};

void ffn_forward(const Tensor& x, const FFNWeights& w, Tensor& out);

// Quantized twin of ffn_forward.
void ffn_forward_q(const Tensor& x, const QuantLayer& ql, Tensor& out);

// One transformer layer: norm -> attention -> residual, then
// norm -> ffn -> residual. The norms sit *before* each sublayer (pre-norm),
// which is what makes deep stacks trainable without warmup tricks.
struct LayerWeights {
    std::vector<float> input_norm;            // before attention
    std::vector<float> post_attention_norm;   // before the ffn
    AttentionWeights attn;
    FFNWeights ffn;
};

void layer_forward(const Tensor& x, const LayerWeights& w,
                   const AttentionConfig& cfg, const RopeTable& rope,
                   Tensor& out);

// The whole model: embedding -> N layers -> final norm -> lm_head.
class Model {
public:
    Model(const std::string& model_dir);

    // ids in, logits out. logits is [seq, vocab].
    void forward(const std::vector<int>& ids, Tensor& logits);

    const ModelConfig& config() const { return cfg_; }

    // Cached generation. Call reset_cache() once, then step() per token.
    void reset_cache(int max_seq);
    void step(int token_id, Tensor& logits);   // logits is [1, vocab]

    // Replaces the f32 projection matrices with int8 + per-row scales, in
    // place, and frees the f32 originals. Norms, embeddings, and lm_head
    // stay f32 -- norms are tiny, embeddings are precision-sensitive.
    void quantize();
    bool is_quantized() const { return quantized_; }
    size_t weight_bytes() const;

private:
    ModelConfig cfg_;
    AttentionConfig acfg_;
    RopeTable rope_;

    std::vector<float> embed_;      // [vocab, hidden], flattened
    std::vector<float> lm_head_;    // [vocab, hidden], flattened
    std::vector<float> final_norm_;
    std::vector<LayerWeights> layers_;
    std::vector<KVCache> caches_;

    bool quantized_ = false;
    std::vector<QuantLayer> qlayers_;
};

}  // namespace verbum