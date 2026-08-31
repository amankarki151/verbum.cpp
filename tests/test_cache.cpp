#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "verbum/attention.h"
#include "verbum/generate.h"
#include "verbum/tensor.h"

using namespace verbum;

static int failures = 0;

static void check(bool cond, const std::string& what) {
    if (cond) {
        std::cout << "  ok    " << what << "\n";
    } else {
        std::cout << "  FAIL  " << what << "\n";
        failures++;
    }
}

int main() {
    std::cout << "kv cache vs full-sequence attention\n";
    // KVH=1 is included on purpose but it can't catch a GQA bug on its own --
    // the KVH=2 cases are the ones doing that work.
    for (auto geom : std::vector<std::array<int, 5>>{
             {6, 16, 4, 2, 8}, {5, 32, 4, 2, 16}, {8, 16, 2, 1, 8}}) {
        const int seq = geom[0], hidden = geom[1];
        const int H = geom[2], KVH = geom[3], D = geom[4];

        std::mt19937 rng(42);
        std::normal_distribution<float> nd(0.0f, 0.3f);

        AttentionConfig cfg;
        cfg.n_heads = H; cfg.n_kv_heads = KVH; cfg.head_dim = D;
        cfg.hidden = hidden; cfg.rms_eps = 1e-6f;

        auto fill = [&](Tensor& t) {
            for (int64_t i = 0; i < t.numel(); i++) t.data()[i] = nd(rng);
        };

        AttentionWeights w;
        w.q_proj = Tensor({static_cast<int64_t>(H) * D, hidden}); fill(w.q_proj);
        w.k_proj = Tensor({static_cast<int64_t>(KVH) * D, hidden}); fill(w.k_proj);
        w.v_proj = Tensor({static_cast<int64_t>(KVH) * D, hidden}); fill(w.v_proj);
        w.o_proj = Tensor({hidden, static_cast<int64_t>(H) * D}); fill(w.o_proj);
        w.q_norm.resize(D); w.k_norm.resize(D);
        for (int i = 0; i < D; i++) {
            w.q_norm[i] = 1.0f + nd(rng) * 0.1f;
            w.k_norm[i] = 1.0f + nd(rng) * 0.1f;
        }

        RopeTable rope = build_rope_table(D, seq + 8, 1e6f);
        Tensor x({seq, hidden}); fill(x);

        Tensor batch_out({seq, hidden});
        attention_forward(x, w, cfg, rope, batch_out);

        KVCache cache;
        cache.init(KVH, D, seq + 8);
        Tensor step_out({seq, hidden});
        for (int t = 0; t < seq; t++) {
            Tensor xi({1, hidden});
            for (int i = 0; i < hidden; i++) xi.data()[i] = x.at(t, i);
            Tensor oi({1, hidden});
            attention_step(xi, w, cfg, rope, cache, oi);
            for (int i = 0; i < hidden; i++) step_out.at(t, i) = oi.data()[i];
        }

        float worst = 0.0f;
        for (int64_t i = 0; i < batch_out.numel(); i++) {
            worst = std::max(worst, std::fabs(batch_out.data()[i] - step_out.data()[i]));
        }
        check(worst < 1e-5f, "seq=" + std::to_string(seq) + " H=" + std::to_string(H) +
                                 " KVH=" + std::to_string(KVH) + " D=" + std::to_string(D) +
                                 " max diff " + std::to_string(worst));
    }

    std::cout << "\nsampler\n";
    {
        std::vector<float> logits = {1.0f, 5.0f, 2.0f, 3.0f};

        SamplerConfig c; c.temperature = 0.0f;
        Sampler s(c);
        check(s.sample(logits.data(), 4) == 1, "temperature 0 gives argmax");

        SamplerConfig c2; c2.temperature = 1.0f; c2.top_k = 1;
        Sampler s2(c2);
        check(s2.sample(logits.data(), 4) == 1, "top_k=1 gives argmax");

        SamplerConfig c3; c3.temperature = 1.0f; c3.top_k = 2; c3.top_p = 1.0f; c3.seed = 7;
        Sampler s3(c3);
        bool only_top2 = true;
        for (int i = 0; i < 300; i++) {
            int t = s3.sample(logits.data(), 4);
            if (t != 1 && t != 3) only_top2 = false;
        }
        check(only_top2, "top_k=2 never picks outside the top 2");

        SamplerConfig c4; c4.temperature = 1.0f; c4.top_k = 4; c4.seed = 123;
        Sampler a(c4), b(c4);
        bool same = true;
        for (int i = 0; i < 50; i++) {
            same &= (a.sample(logits.data(), 4) == b.sample(logits.data(), 4));
        }
        check(same, "same seed gives the same sequence");

        SamplerConfig c5; c5.temperature = 0.05f; c5.top_k = 4; c5.top_p = 1.0f; c5.seed = 1;
        Sampler s5(c5);
        int hits = 0;
        for (int i = 0; i < 200; i++) if (s5.sample(logits.data(), 4) == 1) hits++;
        check(hits > 190, "low temperature concentrates on argmax (" +
                              std::to_string(hits) + "/200)");
    }

    std::cout << "\n" << (failures == 0 ? "all good" : "SOMETHING FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}