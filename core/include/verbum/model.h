#pragma once

#include <string>
#include <vector>

#include "verbum/attention.h"
#include "verbum/config.h"
#include "verbum/nn.h"
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

private:
    ModelConfig cfg_;
    AttentionConfig acfg_;
    RopeTable rope_;

    std::vector<float> embed_;      // [vocab, hidden], flattened
    std::vector<float> lm_head_;    // [vocab, hidden], flattened
    std::vector<float> final_norm_;
    std::vector<LayerWeights> layers_;
};

}  // namespace verbum