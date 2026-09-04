#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "verbum/model.h"
#include "verbum/tokenizer.h"

using namespace verbum;

static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0;
    for (size_t i = 0; i < a.size(); i++) dot += (double)a[i] * b[i];
    return (float)dot;   // already L2-normalised
}

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "models/qwen3-0.6b";

    Tokenizer tok(dir + "/tokenizer.json");
    std::fprintf(stderr, "loading weights...\n");
    Model model(dir);

    const std::vector<std::string> texts = {
        "I lost my sword in the forest",
        "My blade went missing in the woods",
        "The bread at the inn is stale",
        "What time does the market open",
    };

    std::vector<std::vector<float>> vecs;
    for (const auto& t : texts) vecs.push_back(model.embed_text(tok.encode(t)));

    std::printf("\ncosine similarity matrix\n");
    std::printf("%-36s", "");
    for (size_t j = 0; j < texts.size(); j++) std::printf("  [%zu]  ", j);
    std::printf("\n");
    for (size_t i = 0; i < texts.size(); i++) {
        std::printf("[%zu] %-32.32s", i, texts[i].c_str());
        for (size_t j = 0; j < texts.size(); j++) {
            std::printf(" %6.3f", cosine(vecs[i], vecs[j]));
        }
        std::printf("\n");
    }

    const float related = cosine(vecs[0], vecs[1]);
    const float unrelated = cosine(vecs[0], vecs[2]);
    std::printf("\nsword/blade (should be high):   %.3f\n", related);
    std::printf("sword/bread (should be lower):  %.3f\n", unrelated);
    std::printf("\n%s\n", related > unrelated
        ? "embeddings separate topics -- usable for retrieval"
        : "FAIL: unrelated text scored as close as related text");
    return related > unrelated ? 0 : 1;
}