#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "verbum/attention.h"
#include "verbum/tensor.h"

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

int main(int argc, char** argv) {
    std::string ref_path = (argc > 1) ? argv[1] : "tests/reference_attn.json";

    std::cout << "softmax\n";
    {
        Tensor s({1, 3}, {1, 2, 3});
        softmax_rows(s);
        double sum = 0;
        for (int i = 0; i < 3; i++) sum += s.data()[i];
        check(std::fabs(sum - 1.0) < 1e-6, "row sums to 1");
        check(s.data()[2] > s.data()[1] && s.data()[1] > s.data()[0],
              "bigger input stays bigger");

        // The reason for subtracting the max -- without it this row is all nan.
        Tensor big({1, 3}, {1000, 1001, 1002});
        softmax_rows(big);
        bool finite = true;
        double s2 = 0;
        for (int i = 0; i < 3; i++) {
            finite &= std::isfinite(big.data()[i]);
            s2 += big.data()[i];
        }
        check(finite && std::fabs(s2 - 1.0) < 1e-6, "large scores don't overflow");

        Tensor masked({1, 3},
                      {1.0f, -std::numeric_limits<float>::infinity(), 2.0f});
        softmax_rows(masked);
        check(masked.data()[1] == 0.0f, "-inf goes to exactly 0");
    }

    std::cout << "\nagainst the numpy reference\n";
    {
        std::ifstream in(ref_path);
        if (!in) {
            std::cerr << "  no reference file at " << ref_path
                      << " -- run scripts/dump_reference_attn.py first\n";
            failures++;
        } else {
            json ref;
            in >> ref;

            for (const auto& c : ref.at("attention")) {
                const int64_t seq = c.at("seq").get<int64_t>();
                const int64_t hidden = c.at("hidden").get<int64_t>();
                const int64_t H = c.at("H").get<int64_t>();
                const int64_t KVH = c.at("KVH").get<int64_t>();
                const int64_t D = c.at("D").get<int64_t>();

                AttentionConfig cfg;
                cfg.n_heads = static_cast<int>(H);
                cfg.n_kv_heads = static_cast<int>(KVH);
                cfg.head_dim = static_cast<int>(D);
                cfg.hidden = static_cast<int>(hidden);
                cfg.rms_eps = c.at("eps").get<float>();

                AttentionWeights w;
                w.q_proj = Tensor({H * D, hidden}, c.at("wq").get<std::vector<float>>());
                w.k_proj = Tensor({KVH * D, hidden}, c.at("wk").get<std::vector<float>>());
                w.v_proj = Tensor({KVH * D, hidden}, c.at("wv").get<std::vector<float>>());
                w.o_proj = Tensor({hidden, H * D}, c.at("wo").get<std::vector<float>>());
                w.q_norm = c.at("qn").get<std::vector<float>>();
                w.k_norm = c.at("kn").get<std::vector<float>>();

                RopeTable rope = build_rope_table(static_cast<int>(D),
                                                  static_cast<int>(seq) + 8,
                                                  c.at("theta").get<float>());

                Tensor x({seq, hidden}, c.at("x").get<std::vector<float>>());
                Tensor out({seq, hidden});
                attention_forward(x, w, cfg, rope, out);

                auto yv = c.at("y").get<std::vector<float>>();
                float worst = 0.0f;
                for (size_t i = 0; i < yv.size(); i++) {
                    worst = std::max(worst, std::fabs(yv[i] - out.data()[i]));
                }
                check(worst < 1e-5f,
                      "attention seq=" + std::to_string(seq) +
                          " H=" + std::to_string(H) +
                          " KVH=" + std::to_string(KVH) +
                          " D=" + std::to_string(D) +
                          " max diff " + std::to_string(worst));
            }
        }
    }

    std::cout << "\n" << (failures == 0 ? "all good" : "SOMETHING FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}