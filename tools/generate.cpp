#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "verbum/generate.h"
#include "verbum/model.h"
#include "verbum/tokenizer.h"

using namespace verbum;

int main(int argc, char** argv) {
    std::string model_dir = "models/qwen3-0.6b";
    std::string prompt = "The capital of France is";
    int max_tokens = 40;
    SamplerConfig sc;
    sc.temperature = 0.8f;
    sc.top_k = 40;
    sc.top_p = 0.95f;
    sc.seed = 1234;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "-m") model_dir = next();
        else if (a == "-p") prompt = next();
        else if (a == "-n") max_tokens = std::stoi(next());
        else if (a == "-t") sc.temperature = std::stof(next());
        else if (a == "--top-k") sc.top_k = std::stoi(next());
        else if (a == "--top-p") sc.top_p = std::stof(next());
        else if (a == "--seed") sc.seed = std::stoull(next());
        else if (a == "--greedy") sc.temperature = 0.0f;
        else {
            std::cerr << "usage: generate [-m dir] [-p prompt] [-n tokens] "
                         "[-t temp] [--top-k k] [--top-p p] [--seed s] [--greedy]\n";
            return 1;
        }
    }

    try {
        Tokenizer tok(model_dir + "/tokenizer.json");
        std::cerr << "loading weights...\n";
        Model model(model_dir);
        Sampler sampler(sc);

        std::vector<int> ids = tok.encode(prompt);
        if (ids.empty()) {
            std::cerr << "prompt tokenised to nothing\n";
            return 1;
        }

        model.reset_cache(static_cast<int>(ids.size()) + max_tokens + 8);

        Tensor logits({1, static_cast<int64_t>(model.config().vocab_size)});
        const int64_t vocab = model.config().vocab_size;

        // Prefill: push the prompt through one token at a time to fill the
        // cache. Only the logits from the last prompt token matter.
        auto t_start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < ids.size(); i++) {
            model.step(ids[i], logits);
        }
        auto t_prefill = std::chrono::steady_clock::now();

        std::cout << prompt << std::flush;

        // Decode: sample, print, feed back in.
        int generated = 0;
        for (int i = 0; i < max_tokens; i++) {
            int next_id = sampler.sample(logits.data(), vocab);

            std::string piece = tok.decode({next_id}, /*skip_special=*/false);
            // Stop on an end-of-turn / end-of-text special token.
            if (piece.rfind("<|", 0) == 0 && piece.find("|>") != std::string::npos) break;

            std::cout << piece << std::flush;
            generated++;
            model.step(next_id, logits);
        }
        auto t_end = std::chrono::steady_clock::now();

        std::cout << "\n";

        const double prefill_s =
            std::chrono::duration<double>(t_prefill - t_start).count();
        const double decode_s =
            std::chrono::duration<double>(t_end - t_prefill).count();

        std::cerr << "\nprompt: " << ids.size() << " tokens in " << prefill_s << " s ("
                  << (ids.size() / prefill_s) << " tok/s)\n";
        if (generated > 0) {
            std::cerr << "generated: " << generated << " tokens in " << decode_s << " s ("
                      << (generated / decode_s) << " tok/s)\n";
        }
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}