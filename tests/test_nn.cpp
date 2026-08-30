#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "verbum/nn.h"
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

static bool close(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(a) + std::fabs(b));
}

static float max_abs_diff(const std::vector<float>& a, const float* b, size_t n) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; i++) worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

int main(int argc, char** argv) {
    std::string ref_path = (argc > 1) ? argv[1] : "tests/reference_nn.json";

    std::cout << "rmsnorm, hand-computed\n";
    {
        // x = [3, 4] -> mean(x^2) = (9+16)/2 = 12.5, rms = sqrt(12.5)
        Tensor x({1, 2}, {3.0f, 4.0f});
        Tensor out({1, 2});
        rmsnorm(x, {1.0f, 1.0f}, 0.0f, out);
        check(close(out.at(0, 0), 3.0f / std::sqrt(12.5f)), "x[0] scaled by 1/rms");
        check(close(out.at(0, 1), 4.0f / std::sqrt(12.5f)), "x[1] scaled by 1/rms");

        Tensor out2({1, 2});
        rmsnorm(x, {2.0f, 0.5f}, 0.0f, out2);
        check(close(out2.at(0, 0), 2.0f * 3.0f / std::sqrt(12.5f)), "weight applied to x[0]");
        check(close(out2.at(0, 1), 0.5f * 4.0f / std::sqrt(12.5f)), "weight applied to x[1]");

        Tensor u({1, 4}, {5.0f, 5.0f, 5.0f, 5.0f});
        Tensor uo({1, 4});
        rmsnorm(u, {1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, uo);
        bool all_one = true;
        for (int i = 0; i < 4; i++) all_one &= close(uo.data()[i], 1.0f);
        check(all_one, "a constant row normalises to all ones");
    }

    std::cout << "\nrope, structural properties\n";
    {
        RopeTable t = build_rope_table(8, 16, 10000.0f);

        Tensor x({1, 8}, {1, 2, 3, 4, 5, 6, 7, 8});
        Tensor before = x;
        apply_rope(x, 0, t);
        bool same = true;
        for (int i = 0; i < 8; i++) same &= close(x.data()[i], before.data()[i]);
        check(same, "position 0 leaves the vector unchanged");

        Tensor y({1, 8}, {1, 2, 3, 4, 5, 6, 7, 8});
        double before_norm = 0, after_norm = 0;
        for (int i = 0; i < 8; i++) before_norm += y.data()[i] * y.data()[i];
        apply_rope(y, 5, t);
        for (int i = 0; i < 8; i++) after_norm += y.data()[i] * y.data()[i];
        check(std::fabs(before_norm - after_norm) < 1e-3, "rotation preserves vector length");

        Tensor multi({2, 8}, {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8});
        apply_rope(multi, 3, t);
        bool heads_match = true;
        for (int i = 0; i < 8; i++) heads_match &= close(multi.at(0, i), multi.at(1, i));
        check(heads_match, "identical heads rotate identically");
    }

    std::cout << "\nembedding lookup\n";
    {
        std::vector<float> table;
        for (int v = 0; v < 4; v++)
            for (int d = 0; d < 3; d++) table.push_back(v * 10.0f + d);

        Tensor out({2, 3});
        embedding_lookup(table, 3, {2, 0}, out);
        check(close(out.at(0, 0), 20.0f) && close(out.at(0, 2), 22.0f), "row for id 2");
        check(close(out.at(1, 0), 0.0f) && close(out.at(1, 2), 2.0f), "row for id 0");

        bool threw = false;
        Tensor bad({1, 3});
        try { embedding_lookup(table, 3, {99}, bad); }
        catch (const std::exception&) { threw = true; }
        check(threw, "out-of-range id throws");
    }

    std::cout << "\nagainst the numpy reference\n";
    {
        std::ifstream in(ref_path);
        if (!in) {
            std::cerr << "  no reference file at " << ref_path
                      << " -- run scripts/dump_reference_nn.py first\n";
            failures++;
        } else {
            json ref;
            in >> ref;

            for (const auto& c : ref.at("rmsnorm")) {
                const int64_t rows = c.at("rows").get<int64_t>();
                const int64_t dim = c.at("dim").get<int64_t>();
                const float eps = c.at("eps").get<float>();
                auto xv = c.at("x").get<std::vector<float>>();
                auto wv = c.at("w").get<std::vector<float>>();
                auto yv = c.at("y").get<std::vector<float>>();

                Tensor x({rows, dim}, xv);
                Tensor out({rows, dim});
                rmsnorm(x, wv, eps, out);

                float worst = max_abs_diff(yv, out.data(), yv.size());
                check(worst < 1e-5f,
                      "rmsnorm [" + std::to_string(rows) + "," + std::to_string(dim) +
                          "] max diff " + std::to_string(worst));
            }

            for (const auto& c : ref.at("rope")) {
                const int64_t heads = c.at("heads").get<int64_t>();
                const int64_t head_dim = c.at("head_dim").get<int64_t>();
                const float theta = c.at("theta").get<float>();
                const int pos = c.at("pos").get<int>();
                auto xv = c.at("x").get<std::vector<float>>();
                auto yv = c.at("y").get<std::vector<float>>();

                RopeTable t = build_rope_table(static_cast<int>(head_dim), 64, theta);
                Tensor x({heads, head_dim}, xv);
                apply_rope(x, pos, t);

                float worst = max_abs_diff(yv, x.data(), yv.size());
                check(worst < 1e-4f,
                      "rope head_dim=" + std::to_string(head_dim) + " pos=" +
                          std::to_string(pos) + " max diff " + std::to_string(worst));
            }
        }
    }

    std::cout << "\n" << (failures == 0 ? "all good" : "SOMETHING FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}