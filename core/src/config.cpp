#include "verbum/config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace verbum {

using json = nlohmann::json;

namespace {

// Pull a key, blow up loudly if it's missing. A silently-defaulted config
// value turns into a wrong-shaped tensor 300 lines later, which is a much
// worse debugging session than a hard error here.
template <typename T>
T require(const json& j, const char* key) {
    if (!j.contains(key)) {
        throw std::runtime_error(std::string("config.json missing key: ") + key);
    }
    return j.at(key).get<T>();
}

template <typename T>
T optional(const json& j, const char* key, T fallback) {
    if (!j.contains(key) || j.at(key).is_null()) return fallback;
    return j.at(key).get<T>();
}

}  // namespace

ModelConfig ModelConfig::from_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("can't open " + path);

    json j;
    in >> j;

    ModelConfig c;
    c.hidden_size            = require<int>(j, "hidden_size");
    c.num_hidden_layers      = require<int>(j, "num_hidden_layers");
    c.num_attention_heads    = require<int>(j, "num_attention_heads");
    c.intermediate_size      = require<int>(j, "intermediate_size");
    c.vocab_size             = require<int>(j, "vocab_size");

    // Older configs don't have num_key_value_heads. If it's absent the model
    // isn't using grouped-query attention, so kv heads == attention heads.
    c.num_key_value_heads    = optional<int>(j, "num_key_value_heads",
                                             c.num_attention_heads);
    c.max_position_embeddings = optional<int>(j, "max_position_embeddings", 32768);
    c.head_dim               = optional<int>(j, "head_dim",
                                             c.hidden_size / c.num_attention_heads);
    c.rms_norm_eps           = optional<float>(j, "rms_norm_eps", 1e-6f);
    c.rope_theta             = optional<float>(j, "rope_theta", 10000.0f);
    c.tie_word_embeddings    = optional<bool>(j, "tie_word_embeddings", false);

    if (c.num_attention_heads % c.num_key_value_heads != 0) {
        throw std::runtime_error("attention heads must be a multiple of kv heads");
    }
    return c;
}

std::string ModelConfig::describe() const {
    std::ostringstream o;
    o << "hidden_size=" << hidden_size
      << " layers=" << num_hidden_layers
      << " heads=" << num_attention_heads
      << " kv_heads=" << num_key_value_heads
      << " head_dim=" << head_dim
      << " ffn=" << intermediate_size
      << " vocab=" << vocab_size
      << " rope_theta=" << rope_theta
      << " tied_embeddings=" << (tie_word_embeddings ? "yes" : "no");
    return o.str();
}

}  // namespace verbum