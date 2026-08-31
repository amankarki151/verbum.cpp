#include "verbum/model.h"

#include <cmath>
#include <stdexcept>

#include "verbum/ops.h"

namespace verbum {

namespace {

inline float silu(float x) {
    // x * sigmoid(x). Smooth, non-monotonic near zero, and cheap.
    return x / (1.0f + std::exp(-x));
}

// Pulls a tensor out of the file and materialises it as f32 in the shape the
// header says it has. Everything in the model file is bf16, so this is where
// the widening happens.
Tensor load_tensor(const SafeTensors& st, const std::string& name) {
    const TensorView& v = st.get(name);
    return Tensor(v.shape, st.to_float(name));
}

std::vector<float> load_vector(const SafeTensors& st, const std::string& name) {
    return st.to_float(name);
}

}  // namespace

void ffn_forward(const Tensor& x, const FFNWeights& w, Tensor& out) {
    if (x.rank() != 2) throw std::runtime_error("ffn wants a 2-D input");
    const int64_t seq = x.dim(0);
    const int64_t ffn = w.gate_proj.dim(0);

    Tensor gate({seq, ffn});
    Tensor up({seq, ffn});
    matmul_nt(x, w.gate_proj, gate);
    matmul_nt(x, w.up_proj, up);

    // silu on the gate branch, then elementwise into the up branch.
    for (int64_t i = 0; i < gate.numel(); i++) {
        gate.data()[i] = silu(gate.data()[i]) * up.data()[i];
    }

    matmul_nt(gate, w.down_proj, out);
}

void layer_forward(const Tensor& x, const LayerWeights& w,
                   const AttentionConfig& cfg, const RopeTable& rope,
                   Tensor& out) {
    const int64_t seq = x.dim(0);
    const int64_t hidden = x.dim(1);

    Tensor normed({seq, hidden});
    Tensor attn_out({seq, hidden});

    // attention sublayer
    rmsnorm(x, w.input_norm, cfg.rms_eps, normed);
    attention_forward(normed, w.attn, cfg, rope, attn_out);
    for (int64_t i = 0; i < out.numel(); i++) {
        out.data()[i] = x.data()[i] + attn_out.data()[i];
    }

    // ffn sublayer. The residual here comes from `out` -- the result of the
    // attention sublayer -- not from the original x. Getting that wrong runs
    // fine and quietly degrades the model.
    Tensor ffn_out({seq, hidden});
    rmsnorm(out, w.post_attention_norm, cfg.rms_eps, normed);
    ffn_forward(normed, w.ffn, ffn_out);
    for (int64_t i = 0; i < out.numel(); i++) {
        out.data()[i] += ffn_out.data()[i];
    }
}

Model::Model(const std::string& model_dir) {
    cfg_ = ModelConfig::from_file(model_dir + "/config.json");

    acfg_.n_heads = cfg_.num_attention_heads;
    acfg_.n_kv_heads = cfg_.num_key_value_heads;
    acfg_.head_dim = cfg_.head_dim;
    acfg_.hidden = cfg_.hidden_size;
    acfg_.rms_eps = cfg_.rms_norm_eps;

    SafeTensors st(model_dir);

    embed_ = load_vector(st, "model.embed_tokens.weight");
    final_norm_ = load_vector(st, "model.norm.weight");

    // Qwen3-0.6B ties the output head to the input embeddings. The file may
    // still carry an lm_head tensor; prefer it if it's there, otherwise reuse
    // the embedding matrix.
    if (st.has("lm_head.weight")) {
        lm_head_ = load_vector(st, "lm_head.weight");
    } else {
        lm_head_ = embed_;
    }

    const int rope_positions =
        cfg_.max_position_embeddings > 4096 ? 4096 : cfg_.max_position_embeddings;
    rope_ = build_rope_table(cfg_.head_dim, rope_positions, cfg_.rope_theta);

    layers_.resize(cfg_.num_hidden_layers);
    for (int i = 0; i < cfg_.num_hidden_layers; i++) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        LayerWeights& L = layers_[i];

        L.input_norm = load_vector(st, p + "input_layernorm.weight");
        L.post_attention_norm = load_vector(st, p + "post_attention_layernorm.weight");

        L.attn.q_proj = load_tensor(st, p + "self_attn.q_proj.weight");
        L.attn.k_proj = load_tensor(st, p + "self_attn.k_proj.weight");
        L.attn.v_proj = load_tensor(st, p + "self_attn.v_proj.weight");
        L.attn.o_proj = load_tensor(st, p + "self_attn.o_proj.weight");
        L.attn.q_norm = load_vector(st, p + "self_attn.q_norm.weight");
        L.attn.k_norm = load_vector(st, p + "self_attn.k_norm.weight");

        L.ffn.gate_proj = load_tensor(st, p + "mlp.gate_proj.weight");
        L.ffn.up_proj = load_tensor(st, p + "mlp.up_proj.weight");
        L.ffn.down_proj = load_tensor(st, p + "mlp.down_proj.weight");
    }
}

void Model::forward(const std::vector<int>& ids, Tensor& logits) {
    const int64_t seq = static_cast<int64_t>(ids.size());
    const int64_t hidden = cfg_.hidden_size;
    const int64_t vocab = cfg_.vocab_size;

    if (logits.rank() != 2 || logits.dim(0) != seq || logits.dim(1) != vocab) {
        throw std::runtime_error("logits tensor is the wrong shape");
    }

    Tensor h({seq, hidden});
    embedding_lookup(embed_, hidden, ids, h);

    Tensor next({seq, hidden});
    for (int i = 0; i < cfg_.num_hidden_layers; i++) {
        layer_forward(h, layers_[i], acfg_, rope_, next);
        h = next;
    }

    Tensor normed({seq, hidden});
    rmsnorm(h, final_norm_, cfg_.rms_norm_eps, normed);

    Tensor head({vocab, hidden}, lm_head_);
    matmul_nt(normed, head, logits);
}

// One token through one layer, using the cache. Same structure as
// layer_forward -- pre-norm, residual, pre-norm, residual -- just with the
// cached attention step in the middle.
static void layer_step(const Tensor& x, const LayerWeights& w,
                       const AttentionConfig& cfg, const RopeTable& rope,
                       KVCache& cache, Tensor& out) {
    const int64_t hidden = x.dim(1);
    Tensor normed({1, hidden});
    Tensor attn_out({1, hidden});

    rmsnorm(x, w.input_norm, cfg.rms_eps, normed);
    attention_step(normed, w.attn, cfg, rope, cache, attn_out);
    for (int64_t i = 0; i < hidden; i++) {
        out.data()[i] = x.data()[i] + attn_out.data()[i];
    }

    Tensor ffn_out({1, hidden});
    rmsnorm(out, w.post_attention_norm, cfg.rms_eps, normed);
    ffn_forward(normed, w.ffn, ffn_out);
    for (int64_t i = 0; i < hidden; i++) {
        out.data()[i] += ffn_out.data()[i];
    }
}

void Model::reset_cache(int max_seq) {
    caches_.resize(cfg_.num_hidden_layers);
    for (auto& c : caches_) {
        c.init(cfg_.num_key_value_heads, cfg_.head_dim, max_seq);
    }
}

void Model::step(int token_id, Tensor& logits) {
    if (caches_.empty()) throw std::runtime_error("call reset_cache() first");

    const int64_t hidden = cfg_.hidden_size;
    const int64_t vocab = cfg_.vocab_size;

    Tensor h({1, hidden});
    embedding_lookup(embed_, hidden, {token_id}, h);

    Tensor next({1, hidden});
    for (int i = 0; i < cfg_.num_hidden_layers; i++) {
        layer_step(h, layers_[i], acfg_, rope_, caches_[i], next);
        h = next;
    }

    Tensor normed({1, hidden});
    rmsnorm(h, final_norm_, cfg_.rms_norm_eps, normed);

    Tensor head({vocab, hidden}, lm_head_);
    matmul_nt(normed, head, logits);
}

}  // namespace verbum