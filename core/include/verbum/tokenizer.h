#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace verbum {

class Tokenizer {
public:
    // Reads a HuggingFace tokenizer.json.
    explicit Tokenizer(const std::string& path);

    std::vector<int> encode(const std::string& text,
                            bool allow_special = true) const;
    std::string decode(const std::vector<int>& ids,
                       bool skip_special = false) const;

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    int token_to_id(const std::string& tok) const;   // -1 if unknown
    std::string id_to_token(int id) const;

private:
    std::vector<int> encode_ordinary(const std::string& text) const;
    std::vector<std::string> bpe(const std::string& piece) const;

    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> merge_ranks_;   // "a b" -> rank

    std::unordered_map<std::string, int> special_to_id_;
    std::vector<bool> is_special_;

    std::unordered_map<uint8_t, uint32_t> byte_to_uni_;
    std::unordered_map<uint32_t, uint8_t> uni_to_byte_;

    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;
};

}  // namespace verbum