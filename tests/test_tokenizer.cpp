#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "verbum/tokenizer.h"

using json = nlohmann::json;

static void print_ids(const std::vector<int>& v, size_t limit = 24) {
    std::cout << "[";
    for (size_t i = 0; i < v.size() && i < limit; i++) {
        if (i) std::cout << ", ";
        std::cout << v[i];
    }
    if (v.size() > limit) std::cout << ", ...";
    std::cout << "]";
}

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1] : "models/qwen3-0.6b";
    std::string ref_path  = (argc > 2) ? argv[2] : "tests/reference_tokens.json";

    try {
        verbum::Tokenizer tok(model_dir + "/tokenizer.json");
        std::cout << "vocab size: " << tok.vocab_size() << "\n\n";

        std::ifstream in(ref_path);
        if (!in) {
            std::cerr << "no reference file at " << ref_path
                      << " -- run scripts/dump_reference_tokens.py first\n";
            return 1;
        }
        json ref;
        in >> ref;

        int passed = 0, failed = 0;

        for (const auto& c : ref) {
            std::string text = c.at("text").get<std::string>();
            std::vector<int> want = c.at("ids").get<std::vector<int>>();
            std::vector<int> got  = tok.encode(text);

            if (got == want) {
                passed++;
            } else {
                failed++;
                std::cout << "MISMATCH on " << json(text).dump() << "\n";
                std::cout << "  want "; print_ids(want); std::cout << "\n";
                std::cout << "  got  "; print_ids(got);  std::cout << "\n";

                size_t k = 0;
                while (k < want.size() && k < got.size() && want[k] == got[k]) k++;
                std::cout << "  first difference at index " << k << "\n";
                if (k < want.size()) {
                    std::cout << "    expected token " << want[k] << " = "
                              << json(tok.id_to_token(want[k])).dump() << "\n";
                }
                if (k < got.size()) {
                    std::cout << "    produced token " << got[k] << " = "
                              << json(tok.id_to_token(got[k])).dump() << "\n";
                }
                std::cout << "\n";
            }

            std::string back = tok.decode(got);
            if (back != text) {
                std::cout << "ROUND TRIP FAILED for " << json(text).dump() << "\n";
                std::cout << "  got back " << json(back).dump() << "\n\n";
                failed++;
            }
        }

        std::cout << passed << " passed, " << failed << " failed\n";
        return failed == 0 ? 0 : 1;

    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}