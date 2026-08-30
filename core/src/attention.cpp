#include "verbum/attention.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "verbum/ops.h"

namespace verbum {

void softmax_rows(Tensor& x) {
    if (x.rank() != 2) throw std::runtime_error("softmax_rows wants a 2-D tensor");
    const int64_t rows = x.dim(0), cols = x.dim(1);

    for (int64_t r = 0; r < rows; r++) {
        float* row = x.data() + r * cols;

        // Subtract the max before exp. Attention scores can get large, and
        // exp(large) is inf, and inf/inf is nan -- one masked row of nans
        // poisons the whole output. Shifting by the max is mathematically a
        // no-op and keeps everything in range.
        float m = row[0];
        for (int64_t c = 1; c < cols; c++) m = std::max(m, row[c]);

        double sum = 0.0;
        for (int64_t c = 0; c < cols; c++) {
            const float e = std::exp(row[c] - m);
            row[c] = e;
            sum += e;
        }
        const float inv = static_cast<float>(1.0 / sum);
        for (int64_t c = 0; c < cols; c++) row[c] *= inv;
    }
}

void attention_forward(const Tensor& x, const AttentionWeights& w,
                       const AttentionConfig& cfg, const RopeTable& rope,
                       Tensor& out) {
    if (x.rank() != 2 || x.dim(1) != cfg.hidden) {
        throw std::runtime_error("attention input must be [seq, hidden]");
    }
    if (cfg.n_heads % cfg.n_kv_heads != 0) {
        throw std::runtime_error("n_heads must be a multiple of n_kv_heads");
    }
    if (rope.head_dim != cfg.head_dim) {
        throw std::runtime_error("rope table head_dim doesn't match the config");
    }

    const int64_t seq = x.dim(0);
    const int H = cfg.n_heads;
    const int KVH = cfg.n_kv_heads;
    const int D = cfg.head_dim;
    const int group = H / KVH;   // how many query heads share one kv head

    if (seq > rope.max_pos) {
        throw std::runtime_error("sequence is longer than the rope table");
    }

    // ---- project ----
    // Note the widths: q is n_heads*head_dim, which for Qwen3-0.6B is
    // 16*128 = 2048, wider than hidden (1024). head_dim is not
    // hidden/n_heads here -- the model decouples them on purpose, and
    // o_proj brings the width back down at the end.
    Tensor q({seq, static_cast<int64_t>(H) * D});
    Tensor k({seq, static_cast<int64_t>(KVH) * D});
    Tensor v({seq, static_cast<int64_t>(KVH) * D});

    matmul_nt(x, w.q_proj, q);
    matmul_nt(x, w.k_proj, k);
    matmul_nt(x, w.v_proj, v);

    // ---- QK-Norm, then RoPE, per token ----
    // Order matters: Qwen3 normalises each head's q and k vector before the
    // rotation, not after. Swapping the two runs fine and gives wrong numbers.
    Tensor qheads({static_cast<int64_t>(H), D});
    Tensor kheads({static_cast<int64_t>(KVH), D});
    Tensor tmp({1, D});
    Tensor tmp_out({1, D});

    for (int64_t t = 0; t < seq; t++) {
        for (int h = 0; h < H; h++) {
            const float* src = q.data() + t * (static_cast<int64_t>(H) * D) + h * D;
            for (int i = 0; i < D; i++) tmp.data()[i] = src[i];
            rmsnorm(tmp, w.q_norm, cfg.rms_eps, tmp_out);
            for (int i = 0; i < D; i++) qheads.data()[h * D + i] = tmp_out.data()[i];
        }
        apply_rope(qheads, static_cast<int>(t), rope);
        for (int h = 0; h < H; h++) {
            float* dst = q.data() + t * (static_cast<int64_t>(H) * D) + h * D;
            for (int i = 0; i < D; i++) dst[i] = qheads.data()[h * D + i];
        }

        for (int h = 0; h < KVH; h++) {
            const float* src = k.data() + t * (static_cast<int64_t>(KVH) * D) + h * D;
            for (int i = 0; i < D; i++) tmp.data()[i] = src[i];
            rmsnorm(tmp, w.k_norm, cfg.rms_eps, tmp_out);
            for (int i = 0; i < D; i++) kheads.data()[h * D + i] = tmp_out.data()[i];
        }
        apply_rope(kheads, static_cast<int>(t), rope);
        for (int h = 0; h < KVH; h++) {
            float* dst = k.data() + t * (static_cast<int64_t>(KVH) * D) + h * D;
            for (int i = 0; i < D; i++) dst[i] = kheads.data()[h * D + i];
        }
    }

    // ---- attention, per query head ----
    Tensor context({seq, static_cast<int64_t>(H) * D});
    context.zero();

    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
    Tensor scores({1, seq});

    for (int h = 0; h < H; h++) {
        const int kvh = h / group;   // which kv head this query head reads from

        for (int64_t i = 0; i < seq; i++) {
            const float* qi = q.data() + i * (static_cast<int64_t>(H) * D) + h * D;

            // Causal mask: a token can only see itself and what came before.
            // Positions after i get -inf so softmax sends them to exactly 0.
            for (int64_t j = 0; j < seq; j++) {
                if (j > i) {
                    scores.data()[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const float* kj =
                    k.data() + j * (static_cast<int64_t>(KVH) * D) + kvh * D;
                float dot = 0.0f;
                for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
                scores.data()[j] = dot * inv_sqrt_d;
            }

            softmax_rows(scores);

            float* ctx = context.data() + i * (static_cast<int64_t>(H) * D) + h * D;
            for (int64_t j = 0; j <= i; j++) {
                const float p = scores.data()[j];
                if (p == 0.0f) continue;
                const float* vj =
                    v.data() + j * (static_cast<int64_t>(KVH) * D) + kvh * D;
                for (int d = 0; d < D; d++) ctx[d] += p * vj[d];
            }
        }
    }

    // ---- output projection ----
    matmul_nt(context, w.o_proj, out);
}

}  // namespace verbum