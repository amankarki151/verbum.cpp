#include "verbum/model.h"

#include <cmath>
#include <stdexcept>

#include "verbum/ops.h"
#ifdef VERBUM_CUDA
#include "verbum/cuda_backend.h"
#endif

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

    for (int64_t i = 0; i < gate.numel(); i++) {
        gate.data()[i] = silu(gate.data()[i]) * up.data()[i];
    }

    matmul_nt(gate, w.down_proj, out);
}

void ffn_forward_q(const Tensor& x, const QuantLayer& ql, Tensor& out) {
    if (x.rank() != 2) throw std::runtime_error("ffn_forward_q wants a 2-D input");
    const int64_t seq = x.dim(0);
    const int64_t ffn = ql.gate_proj.rows;

    Tensor gate({seq, ffn});
    Tensor up({seq, ffn});
    matmul_nt_q8(x, ql.gate_proj, gate);
    matmul_nt_q8(x, ql.up_proj, up);

    for (int64_t i = 0; i < gate.numel(); i++) {
        gate.data()[i] = silu(gate.data()[i]) * up.data()[i];
    }

    matmul_nt_q8(gate, ql.down_proj, out);
}

void layer_forward(const Tensor& x, const LayerWeights& w,
                   const AttentionConfig& cfg, const RopeTable& rope,
                   Tensor& out) {
    const int64_t seq = x.dim(0);
    const int64_t hidden = x.dim(1);

    Tensor normed({seq, hidden});
    Tensor attn_out({seq, hidden});

    rmsnorm(x, w.input_norm, cfg.rms_eps, normed);
    attention_forward(normed, w.attn, cfg, rope, attn_out);
    for (int64_t i = 0; i < out.numel(); i++) {
        out.data()[i] = x.data()[i] + attn_out.data()[i];
    }

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
    if (quantized_) {
        throw std::runtime_error(
            "forward() has no quantized path yet -- use step() for quantized mode");
    }

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

static void layer_step_q(const Tensor& x, const LayerWeights& w,
                         const QuantLayer& ql, const AttentionConfig& cfg,
                         const RopeTable& rope, KVCache& cache, Tensor& out) {
    const int64_t hidden = x.dim(1);
    Tensor normed({1, hidden});
    Tensor attn_out({1, hidden});

    rmsnorm(x, w.input_norm, cfg.rms_eps, normed);
    attention_step_q(normed, w.attn.q_norm, w.attn.k_norm, ql, cfg, rope, cache, attn_out);
    for (int64_t i = 0; i < hidden; i++) {
        out.data()[i] = x.data()[i] + attn_out.data()[i];
    }

    Tensor ffn_out({1, hidden});
    rmsnorm(out, w.post_attention_norm, cfg.rms_eps, normed);
    ffn_forward_q(normed, ql, ffn_out);
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
#ifdef VERBUM_CUDA
    if (cuda_) { step_cuda(token_id, logits); return; }
#endif
    if (caches_.empty()) throw std::runtime_error("call reset_cache() first");

    const int64_t hidden = cfg_.hidden_size;
    const int64_t vocab = cfg_.vocab_size;

    Tensor h({1, hidden});
    embedding_lookup(embed_, hidden, {token_id}, h);

    Tensor next({1, hidden});
    for (int i = 0; i < cfg_.num_hidden_layers; i++) {
        if (quantized_) {
            layer_step_q(h, layers_[i], qlayers_[i], acfg_, rope_, caches_[i], next);
        } else {
            layer_step(h, layers_[i], acfg_, rope_, caches_[i], next);
        }
        h = next;
    }

    Tensor normed({1, hidden});
    rmsnorm(h, final_norm_, cfg_.rms_norm_eps, normed);

    Tensor head({vocab, hidden}, lm_head_);
    matmul_nt(normed, head, logits);
}

void Model::quantize() {
    if (quantized_) return;

    qlayers_.resize(layers_.size());
    for (size_t i = 0; i < layers_.size(); i++) {
        LayerWeights& L = layers_[i];
        QuantLayer& Q = qlayers_[i];

        Q.q_proj = quantize_rows(L.attn.q_proj);
        Q.k_proj = quantize_rows(L.attn.k_proj);
        Q.v_proj = quantize_rows(L.attn.v_proj);
        Q.o_proj = quantize_rows(L.attn.o_proj);
        Q.gate_proj = quantize_rows(L.ffn.gate_proj);
        Q.up_proj = quantize_rows(L.ffn.up_proj);
        Q.down_proj = quantize_rows(L.ffn.down_proj);

        L.attn.q_proj = Tensor();
        L.attn.k_proj = Tensor();
        L.attn.v_proj = Tensor();
        L.attn.o_proj = Tensor();
        L.ffn.gate_proj = Tensor();
        L.ffn.up_proj = Tensor();
        L.ffn.down_proj = Tensor();
    }
    quantized_ = true;
}

size_t Model::weight_bytes() const {
    size_t total = embed_.size() * sizeof(float) + lm_head_.size() * sizeof(float)
                 + final_norm_.size() * sizeof(float);
    for (size_t i = 0; i < layers_.size(); i++) {
        total += layers_[i].input_norm.size() * sizeof(float);
        total += layers_[i].post_attention_norm.size() * sizeof(float);
        total += layers_[i].attn.q_norm.size() * sizeof(float);
        total += layers_[i].attn.k_norm.size() * sizeof(float);
        if (quantized_) {
            const QuantLayer& Q = qlayers_[i];
            total += Q.q_proj.bytes() + Q.k_proj.bytes() + Q.v_proj.bytes()
                   + Q.o_proj.bytes() + Q.gate_proj.bytes() + Q.up_proj.bytes()
                   + Q.down_proj.bytes();
        } else {
            const LayerWeights& L = layers_[i];
            total += static_cast<size_t>(L.attn.q_proj.numel()) * sizeof(float);
            total += static_cast<size_t>(L.attn.k_proj.numel()) * sizeof(float);
            total += static_cast<size_t>(L.attn.v_proj.numel()) * sizeof(float);
            total += static_cast<size_t>(L.attn.o_proj.numel()) * sizeof(float);
            total += static_cast<size_t>(L.ffn.gate_proj.numel()) * sizeof(float);
            total += static_cast<size_t>(L.ffn.up_proj.numel()) * sizeof(float);
            total += static_cast<size_t>(L.ffn.down_proj.numel()) * sizeof(float);
        }
    }
    return total;
}

void Model::to_cuda() {
#ifndef VERBUM_CUDA
    throw std::runtime_error(
        "built without CUDA -- reconfigure with -DVERBUM_ENABLE_CUDA=ON");
#else
    if (cuda_) return;
    if (!cuda_available()) throw std::runtime_error("no CUDA device found");
    if (quantized_) {
        throw std::runtime_error(
            "CUDA path doesn't support quantized weights yet -- pick one");
    }

    const int64_t hidden = cfg_.hidden_size;
    const int64_t vocab = cfg_.vocab_size;
    const int H = cfg_.num_attention_heads;
    const int KVH = cfg_.num_key_value_heads;
    const int D = cfg_.head_dim;
    const int64_t ffn = cfg_.intermediate_size;

    auto upload_vec = [](const std::vector<float>& v) {
        CudaBuffer b = cuda_alloc(v.size() * sizeof(float));
        cuda_upload(b, v.data(), v.size() * sizeof(float));
        return b;
    };
    auto upload_tensor = [](const Tensor& t) {
        CudaBuffer b = cuda_alloc((size_t)t.numel() * sizeof(float));
        cuda_upload(b, t.data(), (size_t)t.numel() * sizeof(float));
        return b;
    };

    d_embed_ = upload_vec(embed_);
    d_lm_head_ = upload_vec(lm_head_);
    d_final_norm_ = upload_vec(final_norm_);
    d_rope_cos_ = upload_vec(rope_.cos);
    d_rope_sin_ = upload_vec(rope_.sin);

    if (caches_.empty()) throw std::runtime_error("call reset_cache() before to_cuda()");
    const int max_seq = caches_[0].max_seq;
    const size_t cache_bytes = (size_t)max_seq * KVH * D * sizeof(float);

    dlayers_.resize(layers_.size());
    for (size_t i = 0; i < layers_.size(); i++) {
        LayerWeights& L = layers_[i];
        CudaLayer& C = dlayers_[i];
        C.q_proj = upload_tensor(L.attn.q_proj);
        C.k_proj = upload_tensor(L.attn.k_proj);
        C.v_proj = upload_tensor(L.attn.v_proj);
        C.o_proj = upload_tensor(L.attn.o_proj);
        C.gate_proj = upload_tensor(L.ffn.gate_proj);
        C.up_proj = upload_tensor(L.ffn.up_proj);
        C.down_proj = upload_tensor(L.ffn.down_proj);
        C.input_norm = upload_vec(L.input_norm);
        C.post_attention_norm = upload_vec(L.post_attention_norm);
        C.q_norm = upload_vec(L.attn.q_norm);
        C.k_norm = upload_vec(L.attn.k_norm);
        C.kcache = cuda_alloc(cache_bytes);
        C.vcache = cuda_alloc(cache_bytes);
        cuda_zero(C.kcache);
        cuda_zero(C.vcache);
    }

    d_x_        = cuda_alloc(hidden * sizeof(float));
    d_normed_   = cuda_alloc(hidden * sizeof(float));
    d_attn_out_ = cuda_alloc(hidden * sizeof(float));
    d_ffn_out_  = cuda_alloc(hidden * sizeof(float));
    d_q_        = cuda_alloc((size_t)H * D * sizeof(float));
    d_k_        = cuda_alloc((size_t)KVH * D * sizeof(float));
    d_v_        = cuda_alloc((size_t)KVH * D * sizeof(float));
    d_context_  = cuda_alloc((size_t)H * D * sizeof(float));
    d_gate_     = cuda_alloc(ffn * sizeof(float));
    d_up_       = cuda_alloc(ffn * sizeof(float));
    d_logits_   = cuda_alloc(vocab * sizeof(float));

    cuda_sync();
    cuda_ = true;
#endif
}

#ifdef VERBUM_CUDA
void Model::step_cuda(int token_id, Tensor& logits) {
    const int64_t hidden = cfg_.hidden_size;
    const int64_t vocab = cfg_.vocab_size;
    const int H = cfg_.num_attention_heads;
    const int KVH = cfg_.num_key_value_heads;
    const int D = cfg_.head_dim;
    const int64_t ffn = cfg_.intermediate_size;
    const float eps = cfg_.rms_norm_eps;
    const int pos = caches_[0].len;

    cuda_upload(d_x_, embed_.data() + (size_t)token_id * hidden,
                hidden * sizeof(float));

    for (size_t l = 0; l < dlayers_.size(); l++) {
        CudaLayer& C = dlayers_[l];

        cuda_rmsnorm(d_x_, C.input_norm, d_normed_, 1, (int)hidden, eps);

        cuda_matmul_nt(d_normed_, C.q_proj, d_q_, 1, H * D, (int)hidden);
        cuda_matmul_nt(d_normed_, C.k_proj, d_k_, 1, KVH * D, (int)hidden);
        cuda_matmul_nt(d_normed_, C.v_proj, d_v_, 1, KVH * D, (int)hidden);

        cuda_rmsnorm(d_q_, C.q_norm, d_q_, H, D, eps);
        cuda_rmsnorm(d_k_, C.k_norm, d_k_, KVH, D, eps);

        cuda_rope(d_q_, d_rope_cos_, d_rope_sin_, pos, H, D);
        cuda_rope(d_k_, d_rope_cos_, d_rope_sin_, pos, KVH, D);

        cuda_cache_append(C.kcache, C.vcache, d_k_, d_v_, pos, KVH * D);

        cuda_attn_decode(d_q_, C.kcache, C.vcache, d_context_,
                         H, KVH, D, pos + 1);

        cuda_matmul_nt(d_context_, C.o_proj, d_attn_out_, 1, (int)hidden, H * D);
        cuda_add(d_x_, d_attn_out_, (int)hidden);

        cuda_rmsnorm(d_x_, C.post_attention_norm, d_normed_, 1, (int)hidden, eps);
        cuda_matmul_nt(d_normed_, C.gate_proj, d_gate_, 1, (int)ffn, (int)hidden);
        cuda_matmul_nt(d_normed_, C.up_proj, d_up_, 1, (int)ffn, (int)hidden);
        cuda_silu_mul(d_gate_, d_up_, (int)ffn);
        cuda_matmul_nt(d_gate_, C.down_proj, d_ffn_out_, 1, (int)hidden, (int)ffn);
        cuda_add(d_x_, d_ffn_out_, (int)hidden);
    }

    cuda_rmsnorm(d_x_, d_final_norm_, d_normed_, 1, (int)hidden, eps);
    cuda_matmul_nt(d_normed_, d_lm_head_, d_logits_, 1, (int)vocab, (int)hidden);

    cuda_sync();
    cuda_download(logits.data(), d_logits_, vocab * sizeof(float));

    for (auto& c : caches_) c.len++;
}
#endif

}  // namespace verbum
