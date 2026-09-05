#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "verbum/generate.h"
#include "verbum/model.h"
#include "verbum/tokenizer.h"

using namespace verbum;

// Nearest-rank percentile. Verified separately against edge cases and a
// realistic outlier-heavy latency distribution before trusting it here.
static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * v.size())) - 1;
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

struct BenchResult {
    std::string mode;
    double mean_tok_s;
    double p50_ms;
    double p99_ms;
    double weight_mb;
};

// Runs one mode: prefill a short prompt, then generate n_tokens, timing each
// individual step() call so p50/p99 mean something -- a single aggregate
// tok/s number can't tell you whether latency is smooth or spiky.
static BenchResult run_mode(const std::string& model_dir, const std::string& mode,
                            int n_tokens) {
    Model model(model_dir);
    if (mode == "int8") model.quantize();

    Tokenizer tok(model_dir + "/tokenizer.json");
    std::vector<int> ids = tok.encode("The capital of France is");

    model.reset_cache(static_cast<int>(ids.size()) + n_tokens + 8);

#ifdef VERBUM_CUDA
    if (mode == "cuda") model.to_cuda();
#endif

    const int64_t vocab = model.config().vocab_size;
    Tensor logits({1, vocab});

    for (size_t i = 0; i < ids.size(); i++) model.step(ids[i], logits);

    // greedy sampling -- deterministic, removes sampling as a variable so
    // every mode is timed doing exactly the same arithmetic
    SamplerConfig sc;
    sc.temperature = 0.0f;
    Sampler sampler(sc);

    std::vector<double> latencies_ms;
    latencies_ms.reserve(n_tokens);

    for (int i = 0; i < n_tokens; i++) {
        const int next_id = sampler.sample(logits.data(), vocab);
        auto t0 = std::chrono::steady_clock::now();
        model.step(next_id, logits);
        auto t1 = std::chrono::steady_clock::now();
        latencies_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double total_s = 0;
    for (double ms : latencies_ms) total_s += ms / 1000.0;

    BenchResult r;
    r.mode = mode;
    r.mean_tok_s = n_tokens / total_s;
    r.p50_ms = percentile(latencies_ms, 50.0);
    r.p99_ms = percentile(latencies_ms, 99.0);
    r.weight_mb = model.weight_bytes() / 1e6;
    return r;
}

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1] : "models/qwen3-0.6b";
    int n_tokens = (argc > 2) ? std::stoi(argv[2]) : 50;

    std::vector<std::string> modes = {"f32", "int8"};
#ifdef VERBUM_CUDA
    modes.push_back("cuda");
#endif

    std::printf("%-6s %10s %10s %10s %10s\n",
               "mode", "tok/s", "p50 ms", "p99 ms", "weight MB");

    for (const auto& mode : modes) {
        std::fprintf(stderr, "running %s...\n", mode.c_str());
        try {
            BenchResult r = run_mode(model_dir, mode, n_tokens);
            std::printf("%-6s %10.2f %10.2f %10.2f %10.1f\n",
                       r.mode.c_str(), r.mean_tok_s, r.p50_ms, r.p99_ms, r.weight_mb);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "  %s failed: %s\n", mode.c_str(), ex.what());
        }
    }
    return 0;
}