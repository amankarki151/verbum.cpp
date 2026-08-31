#include "verbum/generate.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "verbum/nn.h"
#include "verbum/ops.h"

namespace verbum {

void KVCache::init(int n_kv_heads_, int head_dim_, int max_seq_) {
    n_kv_heads = n_kv_heads_;
    head_dim = head_dim_;
    max_seq = max_seq_;
    len = 0;
    const size_t n = static_cast<size_t>(max_seq) * n_kv_heads * head_dim;
    k.assign(n, 0.0f);
    v.assign(n, 0.0f);
}

void attention_step(const Tensor& x, const AttentionWeights& w,
                    const AttentionConfig& cfg, const RopeTable& rope,
                    KVCache& cache, Tensor& out) {
    if (x.rank() != 2 || x.dim(0) != 1 || x.dim(1) != cfg.hidden) {
        throw std::runtime_error("attention_step wants a [1, hidden] input");
    }
    if (cache.len >= cache.max_seq) {
        throw std::runtime_error("kv cache is full");
    }

    const int H = cfg.n_heads;
    const int KVH = cfg.n_kv_heads;
    const int D = cfg.head_dim;
    const int group = H / KVH;
    const int pos = cache.len;

    Tensor q({1, static_cast<int64_t>(H) * D});
    Tensor kcur({1, static_cast<int64_t>(KVH) * D});
    Tensor vcur({1, static_cast<int64_t>(KVH) * D});

    matmul_nt(x, w.q_proj, q);
    matmul_nt(x, w.k_proj, kcur);
    matmul_nt(x, w.v_proj, vcur);

    // QK-Norm per head, then RoPE at this token's position -- same order as
    // the batch path, just for one token.
    Tensor qheads({static_cast<int64_t>(H), D});
    Tensor kheads({static_cast<int64_t>(KVH), D});
    Tensor tmp({1, D}), tmp_out({1, D});

    for (int h = 0; h < H; h++) {
        for (int i = 0; i < D; i++) tmp.data()[i] = q.data()[h * D + i];
        rmsnorm(tmp, w.q_norm, cfg.rms_eps, tmp_out);
        for (int i = 0; i < D; i++) qheads.data()[h * D + i] = tmp_out.data()[i];
    }
    apply_rope(qheads, pos, rope);

    for (int h = 0; h < KVH; h++) {
        for (int i = 0; i < D; i++) tmp.data()[i] = kcur.data()[h * D + i];
        rmsnorm(tmp, w.k_norm, cfg.rms_eps, tmp_out);
        for (int i = 0; i < D; i++) kheads.data()[h * D + i] = tmp_out.data()[i];
    }
    apply_rope(kheads, pos, rope);

    // Append this token's k and v. Everything before it stays untouched --
    // that's the whole point.
    const size_t row = static_cast<size_t>(pos) * KVH * D;
    for (int i = 0; i < KVH * D; i++) {
        cache.k[row + i] = kheads.data()[i];
        cache.v[row + i] = vcur.data()[i];
    }
    cache.len++;

    // Attend over everything in the cache. No mask needed: the cache only
    // holds tokens at or before this position, so causality is structural
    // rather than something we have to enforce with -inf.
    const int n = cache.len;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
    Tensor context({1, static_cast<int64_t>(H) * D});
    context.zero();
    std::vector<float> scores(static_cast<size_t>(n));

    for (int h = 0; h < H; h++) {
        const int kvh = h / group;
        const float* qh = qheads.data() + h * D;

        float m = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < n; j++) {
            const float* kj = cache.k.data() + static_cast<size_t>(j) * KVH * D + kvh * D;
            float dot = 0.0f;
            for (int d = 0; d < D; d++) dot += qh[d] * kj[d];
            scores[j] = dot * inv_sqrt_d;
            m = std::max(m, scores[j]);
        }

        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            scores[j] = std::exp(scores[j] - m);
            sum += scores[j];
        }
        const float inv = static_cast<float>(1.0 / sum);

        float* ctx = context.data() + h * D;
        for (int j = 0; j < n; j++) {
            const float p = scores[j] * inv;
            const float* vj = cache.v.data() + static_cast<size_t>(j) * KVH * D + kvh * D;
            for (int d = 0; d < D; d++) ctx[d] += p * vj[d];
        }
    }

    matmul_nt(context, w.o_proj, out);
}

Sampler::Sampler(const SamplerConfig& cfg) : cfg_(cfg), rng_(cfg.seed) {}

int Sampler::sample(const float* logits, int64_t vocab) {
    // Greedy: no sorting, no randomness, just the biggest logit.
    if (cfg_.temperature <= 0.0f || cfg_.top_k == 1) {
        int best = 0;
        for (int64_t i = 1; i < vocab; i++) {
            if (logits[i] > logits[best]) best = static_cast<int>(i);
        }
        return best;
    }

    scratch_.clear();
    scratch_.reserve(static_cast<size_t>(vocab));
    for (int64_t i = 0; i < vocab; i++) {
        scratch_.emplace_back(logits[i], static_cast<int>(i));
    }

    // Keep only the top-k candidates. partial_sort is enough -- we don't care
    // about the order of everything we're about to throw away.
    size_t keep = scratch_.size();
    if (cfg_.top_k > 0 && static_cast<size_t>(cfg_.top_k) < keep) {
        keep = static_cast<size_t>(cfg_.top_k);
    }
    std::partial_sort(scratch_.begin(), scratch_.begin() + keep, scratch_.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    scratch_.resize(keep);

    // Softmax over the survivors, at temperature.
    const float m = scratch_[0].first;
    double sum = 0.0;
    for (auto& p : scratch_) {
        p.first = std::exp((p.first - m) / cfg_.temperature);
        sum += p.first;
    }
    for (auto& p : scratch_) p.first = static_cast<float>(p.first / sum);

    // top-p (nucleus): walk down the sorted list until the probability mass
    // reaches p, then drop the rest. Cuts the long tail of junk tokens that
    // top-k alone leaves in when the distribution is confident.
    if (cfg_.top_p < 1.0f) {
        double cum = 0.0;
        size_t cut = scratch_.size();
        for (size_t i = 0; i < scratch_.size(); i++) {
            cum += scratch_[i].first;
            if (cum >= cfg_.top_p) { cut = i + 1; break; }
        }
        scratch_.resize(cut);

        double renorm = 0.0;
        for (const auto& p : scratch_) renorm += p.first;
        for (auto& p : scratch_) p.first = static_cast<float>(p.first / renorm);
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng_);
    double cum = 0.0;
    for (const auto& p : scratch_) {
        cum += p.first;
        if (r <= cum) return p.second;
    }
    return scratch_.back().second;
}

}  // namespace verbum