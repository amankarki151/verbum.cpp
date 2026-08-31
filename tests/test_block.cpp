#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "verbum/model.h"

using json = nlohmann::json;
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

static std::vector<float> vec(const json& j, const char* k) {
    return j.at(k).get<std::vector<float>>();
}

int main(int argc, char** argv) {
    std::string ref_path = (argc > 1) ? argv[1] : "tests/reference_block.json";

    std::ifstream in(ref_path);
    if (!in) {
        std::cerr << "no reference file at " << ref_path
                  << " -- run scripts/dump_reference_block.py first\n";
        return 1;
    }
    json ref;
    in >> ref;

    std::cout << "swiglu feed-forward\n";
    for (const auto& c : ref.at("ffn")) {
        const int64_t seq = c.at("seq").get<int64_t>();
        const int64_t hidden = c.at("hidden").get<int64_t>();
        const int64_t f = c.at("ffn").get<int64_t>();

        FFNWeights w;
        w.gate_proj = Tensor({f, hidden}, vec(c, "g"));
        w.up_proj = Tensor({f, hidden}, vec(c, "u"));
        w.down_proj = Tensor({hidden, f}, vec(c, "d"));

        Tensor x({seq, hidden}, vec(c, "x"));
        Tensor out({seq, hidden});
        ffn_forward(x, w, out);

        auto yv = vec(c, "y");
        float worst = 0.0f;
        for (size_t i = 0; i < yv.size(); i++) {
            worst = std::max(worst, std::fabs(yv[i] - out.data()[i]));
        }
        check(worst < 1e-5f, "ffn hidden=" + std::to_string(hidden) +
                                 " max diff " + std::to_string(worst));
    }

    std::cout << "\nfull transformer layer\n";
    for (const auto& c : ref.at("layer")) {
        const int64_t seq = c.at("seq").get<int64_t>();
        const int64_t hidden = c.at("hidden").get<int64_t>();
        const int64_t H = c.at("H").get<int64_t>();
        const int64_t KVH = c.at("KVH").get<int64_t>();
        const int64_t D = c.at("D").get<int64_t>();
        const int64_t F = c.at("F").get<int64_t>();

        AttentionConfig cfg;
        cfg.n_heads = static_cast<int>(H);
        cfg.n_kv_heads = static_cast<int>(KVH);
        cfg.head_dim = static_cast<int>(D);
        cfg.hidden = static_cast<int>(hidden);
        cfg.rms_eps = c.at("eps").get<float>();

        LayerWeights L;
        L.input_norm = vec(c, "inorm");
        L.post_attention_norm = vec(c, "pnorm");
        L.attn.q_proj = Tensor({H * D, hidden}, vec(c, "wq"));
        L.attn.k_proj = Tensor({KVH * D, hidden}, vec(c, "wk"));
        L.attn.v_proj = Tensor({KVH * D, hidden}, vec(c, "wv"));
        L.attn.o_proj = Tensor({hidden, H * D}, vec(c, "wo"));
        L.attn.q_norm = vec(c, "qn");
        L.attn.k_norm = vec(c, "kn");
        L.ffn.gate_proj = Tensor({F, hidden}, vec(c, "g"));
        L.ffn.up_proj = Tensor({F, hidden}, vec(c, "u"));
        L.ffn.down_proj = Tensor({hidden, F}, vec(c, "d"));

        RopeTable rope = build_rope_table(static_cast<int>(D),
                                          static_cast<int>(seq) + 8,
                                          c.at("theta").get<float>());

        Tensor x({seq, hidden}, vec(c, "x"));
        Tensor out({seq, hidden});
        layer_forward(x, L, cfg, rope, out);

        auto yv = vec(c, "y");
        float worst = 0.0f;
        for (size_t i = 0; i < yv.size(); i++) {
            worst = std::max(worst, std::fabs(yv[i] - out.data()[i]));
        }
        check(worst < 1e-5f, "layer seq=" + std::to_string(seq) +
                                 " hidden=" + std::to_string(hidden) +
                                 " max diff " + std::to_string(worst));
    }

    std::cout << "\n" << (failures == 0 ? "all good" : "SOMETHING FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}