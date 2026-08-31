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

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1] : "models/qwen3-0.6b";
    std::string ref_path = (argc > 2) ? argv[2] : "tests/reference_logits.json";

    std::ifstream in(ref_path);
    if (!in) {
        std::cerr << "no reference file at " << ref_path
                  << " -- run scripts/dump_reference_logits.py first\n";
        return 1;
    }
    json ref;
    in >> ref;

    std::cout << "loading model (this takes a moment, everything widens to f32)\n";
    Model model(model_dir);
    std::cout << "loaded: " << model.config().describe() << "\n\n";

    int failures = 0;

    for (const auto& c : ref) {
        const std::string text = c.at("text").get<std::string>();
        auto ids = c.at("ids").get<std::vector<int>>();
        auto top_ids = c.at("top_ids").get<std::vector<int>>();
        auto top_logits = c.at("top_logits").get<std::vector<float>>();
        const int want_argmax = c.at("argmax").get<int>();

        Tensor logits({static_cast<int64_t>(ids.size()),
                       static_cast<int64_t>(model.config().vocab_size)});
        model.forward(ids, logits);

        const int64_t vocab = model.config().vocab_size;
        const float* last = logits.data() + (static_cast<int64_t>(ids.size()) - 1) * vocab;

        // does our argmax match theirs
        int got_argmax = 0;
        for (int64_t i = 1; i < vocab; i++) {
            if (last[i] > last[got_argmax]) got_argmax = static_cast<int>(i);
        }

        // how far off are the top-k logit values
        float worst = 0.0f;
        for (size_t i = 0; i < top_ids.size(); i++) {
            worst = std::max(worst, std::fabs(last[top_ids[i]] - top_logits[i]));
        }

        std::cout << "\"" << text << "\"\n";
        std::cout << "  argmax: want " << want_argmax << ", got " << got_argmax
                  << (got_argmax == want_argmax ? "  ok" : "  MISMATCH") << "\n";
        std::cout << "  top-" << top_ids.size() << " max logit diff: " << worst
                  << (worst < 0.05f ? "  ok" : "  TOO BIG") << "\n\n";

        if (got_argmax != want_argmax) failures++;
        if (worst >= 0.05f) failures++;
    }

    std::cout << (failures == 0 ? "all good -- the forward pass is correct"
                                : "SOMETHING FAILED")
              << "\n";
    return failures == 0 ? 0 : 1;
}