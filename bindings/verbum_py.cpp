#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "verbum/generate.h"
#include "verbum/model.h"
#include "verbum/tokenizer.h"

namespace py = pybind11;
using namespace verbum;

// Bundles the model and tokenizer together, since Python never wants one
// without the other, and exposes text-in/text-out rather than token ids.
// The C++ side keeps its clean separation; this is a convenience layer.
class Engine {
public:
    Engine(const std::string& model_dir, bool quantize)
        : tok_(model_dir + "/tokenizer.json"), model_(model_dir) {
        if (quantize) model_.quantize();
    }

    std::vector<int> encode(const std::string& text) const {
        return tok_.encode(text);
    }

    std::string decode(const std::vector<int>& ids) const {
        return tok_.decode(ids, /*skip_special=*/false);
    }

    std::vector<float> embed_text(const std::string& text) {
        return model_.embed_text(tok_.encode(text));
    }

    // Prompt in, generated text out. Handles prefill, decode, and the stop
    // condition so Python doesn't have to think about the KV-cache at all.
    std::string generate(const std::string& prompt, int max_tokens,
                         float temperature, int top_k, float top_p,
                         uint64_t seed) {
        std::vector<int> ids = tok_.encode(prompt);
        if (ids.empty()) throw std::runtime_error("prompt tokenised to nothing");

        model_.reset_cache(static_cast<int>(ids.size()) + max_tokens + 8);

        SamplerConfig sc;
        sc.temperature = temperature;
        sc.top_k = top_k;
        sc.top_p = top_p;
        sc.seed = seed;
        Sampler sampler(sc);

        const int64_t vocab = model_.config().vocab_size;
        Tensor logits({1, vocab});

        for (size_t i = 0; i < ids.size(); i++) model_.step(ids[i], logits);

        std::string out;
        for (int i = 0; i < max_tokens; i++) {
            const int next_id = sampler.sample(logits.data(), vocab);
            const std::string piece = tok_.decode({next_id}, false);
            if (piece.rfind("<|", 0) == 0 && piece.find("|>") != std::string::npos) break;
            out += piece;
            model_.step(next_id, logits);
        }
        return out;
    }

    int vocab_size() const { return model_.config().vocab_size; }
    int hidden_size() const { return model_.config().hidden_size; }
    bool is_quantized() const { return model_.is_quantized(); }
    size_t weight_bytes() const { return model_.weight_bytes(); }

private:
    Tokenizer tok_;
    Model model_;
};

PYBIND11_MODULE(verbum, m) {
    m.doc() = "verbum.cpp -- an LLM inference engine written from scratch";

    py::class_<Engine>(m, "Engine")
        .def(py::init<const std::string&, bool>(),
             py::arg("model_dir"), py::arg("quantize") = false,
             "Load a model. Set quantize=True for int8 weights.")
        .def("encode", &Engine::encode, py::arg("text"))
        .def("decode", &Engine::decode, py::arg("ids"))
        .def("embed_text", &Engine::embed_text, py::arg("text"),
             "Mean-pooled hidden state as a sentence embedding.")
        .def("generate", &Engine::generate,
             py::arg("prompt"),
             py::arg("max_tokens") = 60,
             py::arg("temperature") = 0.8f,
             py::arg("top_k") = 40,
             py::arg("top_p") = 0.95f,
             py::arg("seed") = 0,
             // Generation is slow and holds no Python state -- releasing the
             // GIL means a UI thread stays responsive while it runs.
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("vocab_size", &Engine::vocab_size)
        .def_property_readonly("hidden_size", &Engine::hidden_size)
        .def_property_readonly("is_quantized", &Engine::is_quantized)
        .def_property_readonly("weight_bytes", &Engine::weight_bytes);
}