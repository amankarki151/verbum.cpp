#pragma once

#include <string>

namespace verbum {

// Everything we need out of the model's config.json. Field names match the
// HuggingFace keys so it's obvious where each one comes from.
struct ModelConfig {
    int hidden_size        = 0;
    int num_hidden_layers  = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;   // < num_attention_heads means grouped-query
    int intermediate_size  = 0;    // feed-forward width
    int vocab_size         = 0;
    int max_position_embeddings = 0;
    int head_dim           = 0;    // usually hidden_size / num_attention_heads
    float rms_norm_eps     = 1e-6f;
    float rope_theta       = 10000.0f;
    bool tie_word_embeddings = false;

    static ModelConfig from_file(const std::string& path);
    std::string describe() const;
};

}  // namespace verbum