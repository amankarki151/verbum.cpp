#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "verbum/config.h"
#include "verbum/safetensors.h"

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "models/qwen3-0.6b";

    try {
        auto cfg = verbum::ModelConfig::from_file(dir + "/config.json");
        std::cout << "config: " << cfg.describe() << "\n\n";

        verbum::SafeTensors st(dir);
        std::cout << "loaded " << st.size() << " tensors\n\n";

        // Print the first handful so you can eyeball the naming scheme.
        auto names = st.names();
        std::cout << "first 8 tensor names:\n";
        for (size_t i = 0; i < names.size() && i < 8; i++) {
            const auto& t = st.get(names[i]);
            std::cout << "  " << t.name << "  " << dtype_name(t.dtype)
                      << "  " << t.shape_string() << "\n";
        }

        // The embedding table is the one tensor whose shape we can predict
        // exactly from the config, so it's the best thing to assert on.
        const char* embed = "model.embed_tokens.weight";
        if (!st.has(embed)) {
            std::cerr << "\nFAIL: expected a tensor called " << embed << "\n";
            return 1;
        }
        const auto& e = st.get(embed);
        std::cout << "\n" << embed << " -> " << e.shape_string() << "\n";

        if (e.shape.size() != 2 ||
            e.shape[0] != cfg.vocab_size ||
            e.shape[1] != cfg.hidden_size) {
            std::cerr << "FAIL: embedding shape doesn't match config "
                      << "(want [" << cfg.vocab_size << ", "
                      << cfg.hidden_size << "])\n";
            return 1;
        }

        // Convert one row and check the numbers aren't garbage. If the header
        // parse were off by even a byte this comes out as inf/nan.
        auto row = st.to_float(embed);
        float lo = row[0], hi = row[0];
        double sum = 0;
        for (size_t i = 0; i < 1000 && i < row.size(); i++) {
            lo = std::min(lo, row[i]);
            hi = std::max(hi, row[i]);
            sum += row[i];
        }
        std::cout << "first 1000 embedding values: min=" << lo
                  << " max=" << hi << " mean=" << (sum / 1000) << "\n";

        if (!std::isfinite(lo) || !std::isfinite(hi)) {
            std::cerr << "FAIL: non-finite values in embeddings\n";
            return 1;
        }
        if (std::abs(lo) > 100 || std::abs(hi) > 100) {
            std::cerr << "FAIL: embedding values are implausibly large, "
                         "header parse is probably off\n";
            return 1;
        }

        std::cout << "\nall good\n";
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}