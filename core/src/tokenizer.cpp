#include "verbum/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace verbum {

using json = nlohmann::json;

namespace {

std::vector<uint32_t> utf8_decode(const std::string& s) {
    std::vector<uint32_t> cps;
    cps.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp;
        int len;
        if (c < 0x80)            { cp = c;        len = 1; }
        else if ((c >> 5) == 0x6){ cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE){ cp = c & 0x0F; len = 3; }
        else                     { cp = c & 0x07; len = 4; }
        for (int k = 1; k < len && i + k < s.size(); k++) {
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        }
        cps.push_back(cp);
        i += len;
    }
    return cps;
}

std::string utf8_encode_one(uint32_t cp) {
    std::string o;
    if (cp < 0x80) {
        o += static_cast<char>(cp);
    } else if (cp < 0x800) {
        o += static_cast<char>(0xC0 | (cp >> 6));
        o += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        o += static_cast<char>(0xE0 | (cp >> 12));
        o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        o += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        o += static_cast<char>(0xF0 | (cp >> 18));
        o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        o += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return o;
}

// Anything non-ASCII gets treated as a letter. That's not what \p{L} means
// strictly -- it lumps CJK punctuation and symbols in with letters -- but it
// covers ordinary text, and the test in step 26 will tell us if it bites.
inline bool is_letter(uint32_t cp) {
    if (cp < 128) return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
    return true;
}
inline bool is_digit(uint32_t cp) { return cp >= '0' && cp <= '9'; }
inline bool is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' ||
           cp == '\r' || cp == 0x0B || cp == 0x0C;
}

// Hand-rolled version of the Qwen pretokenizer pattern:
//   's|'t|'re|'ve|'m|'ll|'d
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}
//   |  ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
//
// std::regex can't do \p{L} or lookahead reliably, and pulling in PCRE2 for
// one pattern isn't worth it, so this walks the string directly. Order of the
// branches matters -- it mirrors the alternation order above.
std::vector<std::string> pretokenize(const std::string& text) {
    auto cp = utf8_decode(text);
    std::vector<std::string> out;
    size_t i = 0, n = cp.size();

    auto emit = [&](size_t a, size_t b) {
        std::string s;
        for (size_t k = a; k < b; k++) s += utf8_encode_one(cp[k]);
        if (!s.empty()) out.push_back(std::move(s));
    };

    while (i < n) {
        // contractions
        if (cp[i] == '\'' && i + 1 < n) {
            char c1 = static_cast<char>(std::tolower(static_cast<int>(cp[i + 1])));
            char c2 = (i + 2 < n)
                        ? static_cast<char>(std::tolower(static_cast<int>(cp[i + 2])))
                        : '\0';
            bool three = (c1 == 'r' && c2 == 'e') ||
                         (c1 == 'v' && c2 == 'e') ||
                         (c1 == 'l' && c2 == 'l');
            bool two   = (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd');
            if (three)     { emit(i, i + 3); i += 3; continue; }
            else if (two)  { emit(i, i + 2); i += 2; continue; }
        }

        // optional single non-letter/non-digit prefix, then a run of letters
        {
            size_t j = i;
            bool ok = true;
            if (!is_letter(cp[j]) && !is_digit(cp[j]) &&
                cp[j] != '\n' && cp[j] != '\r') {
                if (j + 1 < n && is_letter(cp[j + 1])) j++;
                else ok = false;
            }
            if (ok && j < n && is_letter(cp[j])) {
                size_t k = j;
                while (k < n && is_letter(cp[k])) k++;
                emit(i, k);
                i = k;
                continue;
            }
        }

        // single digit -- Qwen splits numbers digit by digit
        if (is_digit(cp[i])) { emit(i, i + 1); i++; continue; }

        // optional leading space, run of symbols, trailing newlines
        {
            size_t j = i;
            if (cp[j] == ' ' && j + 1 < n && !is_space(cp[j + 1]) &&
                !is_letter(cp[j + 1]) && !is_digit(cp[j + 1])) {
                j++;
            }
            if (j < n && !is_space(cp[j]) && !is_letter(cp[j]) && !is_digit(cp[j])) {
                size_t k = j;
                while (k < n && !is_space(cp[k]) && !is_letter(cp[k]) &&
                       !is_digit(cp[k])) {
                    k++;
                }
                while (k < n && (cp[k] == '\n' || cp[k] == '\r')) k++;
                emit(i, k);
                i = k;
                continue;
            }
        }

        // whitespace
        if (is_space(cp[i])) {
            size_t k = i;
            while (k < n && is_space(cp[k])) k++;

            // \s*[\r\n]+ wins if the run contains any newline
            size_t last_nl = std::string::npos;
            for (size_t q = i; q < k; q++) {
                if (cp[q] == '\n' || cp[q] == '\r') last_nl = q;
            }
            if (last_nl != std::string::npos) {
                emit(i, last_nl + 1);
                i = last_nl + 1;
                continue;
            }

            // \s+(?!\S) -- if real text follows, hand the last space to it,
            // since " word" is a single token in a byte-level BPE vocab
            if (k < n && k - i > 1) { emit(i, k - 1); i = k - 1; }
            else                    { emit(i, k);     i = k;     }
            continue;
        }

        emit(i, i + 1);
        i++;
    }
    return out;
}

}  // namespace

Tokenizer::Tokenizer(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("can't open " + path);

    json j;
    in >> j;

    if (!j.contains("model")) throw std::runtime_error("tokenizer.json has no 'model'");
    const json& model = j.at("model");

    // ---- vocab ----
    const json& vocab = model.at("vocab");
    int max_id = -1;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        int id = it.value().get<int>();
        token_to_id_[it.key()] = id;
        max_id = std::max(max_id, id);
    }

    // ---- added / special tokens ----
    if (j.contains("added_tokens")) {
        for (const auto& t : j.at("added_tokens")) {
            std::string content = t.at("content").get<std::string>();
            int id = t.at("id").get<int>();
            token_to_id_[content] = id;
            special_to_id_[content] = id;
            max_id = std::max(max_id, id);
        }
    }

    id_to_token_.assign(max_id + 1, std::string());
    is_special_.assign(max_id + 1, false);
    for (const auto& kv : token_to_id_) {
        if (kv.second >= 0 && kv.second <= max_id) id_to_token_[kv.second] = kv.first;
    }
    for (const auto& kv : special_to_id_) {
        if (kv.second >= 0 && kv.second <= max_id) is_special_[kv.second] = true;
    }

    // ---- merges ----
    // Older files store these as "a b" strings, newer ones as ["a", "b"].
    // Handle both so this doesn't break when you swap models.
    if (model.contains("merges")) {
        const json& merges = model.at("merges");
        int rank = 0;
        for (const auto& m : merges) {
            std::string key;
            if (m.is_string()) {
                key = m.get<std::string>();
            } else if (m.is_array() && m.size() == 2) {
                key = m[0].get<std::string>() + " " + m[1].get<std::string>();
            } else {
                continue;
            }
            merge_ranks_.emplace(key, rank++);
        }
    }

    // ---- byte <-> unicode table ----
    // Same trick GPT-2 used: map all 256 byte values onto printable codepoints
    // so the vocab is pure text and arbitrary binary still round-trips.
    std::vector<int> bs, cs;
    for (int b = '!';  b <= '~';  b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    cs = bs;
    int extra = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + extra);
            extra++;
        }
    }
    for (size_t i = 0; i < bs.size(); i++) {
        byte_to_uni_[static_cast<uint8_t>(bs[i])] = static_cast<uint32_t>(cs[i]);
        uni_to_byte_[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
    }
}

int Tokenizer::token_to_id(const std::string& tok) const {
    auto it = token_to_id_.find(tok);
    return it == token_to_id_.end() ? -1 : it->second;
}

std::string Tokenizer::id_to_token(int id) const {
    if (id < 0 || id >= static_cast<int>(id_to_token_.size())) return {};
    return id_to_token_[id];
}

std::vector<std::string> Tokenizer::bpe(const std::string& piece) const {
    auto cached = bpe_cache_.find(piece);
    if (cached != bpe_cache_.end()) return cached->second;

    // start from single characters of the byte-encoded piece
    std::vector<std::string> parts;
    for (uint32_t c : utf8_decode(piece)) parts.push_back(utf8_encode_one(c));

    while (parts.size() >= 2) {
        int best = std::numeric_limits<int>::max();
        size_t best_i = 0;
        bool found = false;

        for (size_t i = 0; i + 1 < parts.size(); i++) {
            auto it = merge_ranks_.find(parts[i] + " " + parts[i + 1]);
            if (it != merge_ranks_.end() && it->second < best) {
                best = it->second;
                best_i = i;
                found = true;
            }
        }
        if (!found) break;

        std::vector<std::string> merged;
        merged.reserve(parts.size() - 1);
        for (size_t i = 0; i < parts.size();) {
            if (i == best_i) { merged.push_back(parts[i] + parts[i + 1]); i += 2; }
            else             { merged.push_back(parts[i]); i++; }
        }
        parts.swap(merged);
    }

    bpe_cache_[piece] = parts;
    return parts;
}

std::vector<int> Tokenizer::encode_ordinary(const std::string& text) const {
    std::vector<int> ids;
    for (const auto& piece : pretokenize(text)) {
        // raw bytes -> printable codepoints
        std::string mapped;
        for (unsigned char b : piece) {
            mapped += utf8_encode_one(byte_to_uni_.at(b));
        }
        for (const auto& tok : bpe(mapped)) {
            int id = token_to_id(tok);
            if (id < 0) {
                throw std::runtime_error("token not in vocab: " + tok);
            }
            ids.push_back(id);
        }
    }
    return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool allow_special) const {
    if (!allow_special || special_to_id_.empty()) return encode_ordinary(text);

    // Cut the text around any special token before normal encoding, otherwise
    // "<|im_start|>" gets chewed up into a dozen ordinary tokens.
    std::vector<int> ids;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best_at = std::string::npos;
        size_t best_len = 0;
        int best_id = -1;

        for (const auto& kv : special_to_id_) {
            size_t at = text.find(kv.first, pos);
            if (at == std::string::npos) continue;
            // earliest wins; on a tie the longer token wins
            if (at < best_at || (at == best_at && kv.first.size() > best_len)) {
                best_at = at;
                best_len = kv.first.size();
                best_id = kv.second;
            }
        }

        if (best_at == std::string::npos) {
            auto rest = encode_ordinary(text.substr(pos));
            ids.insert(ids.end(), rest.begin(), rest.end());
            break;
        }
        if (best_at > pos) {
            auto chunk = encode_ordinary(text.substr(pos, best_at - pos));
            ids.insert(ids.end(), chunk.begin(), chunk.end());
        }
        ids.push_back(best_id);
        pos = best_at + best_len;
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) continue;

        bool special = is_special_[id];
        if (special && skip_special) continue;

        const std::string& tok = id_to_token_[id];
        if (special) {
            out += tok;          // specials are literal text, not byte-encoded
            continue;
        }
        for (uint32_t c : utf8_decode(tok)) {
            auto it = uni_to_byte_.find(c);
            if (it != uni_to_byte_.end()) out += static_cast<char>(it->second);
        }
    }
    return out;
}

}  // namespace verbum