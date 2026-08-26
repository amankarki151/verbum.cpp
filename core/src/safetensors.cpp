#include "verbum/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace verbum {

using json = nlohmann::json;
namespace fs = std::filesystem;

const char* dtype_name(DType d) {
    switch (d) {
        case DType::F32:  return "F32";
        case DType::F16:  return "F16";
        case DType::BF16: return "BF16";
        case DType::I64:  return "I64";
        case DType::I32:  return "I32";
        case DType::I8:   return "I8";
        case DType::U8:   return "U8";
        case DType::BOOL: return "BOOL";
        default:          return "?";
    }
}

size_t dtype_size(DType d) {
    switch (d) {
        case DType::F32: case DType::I32: return 4;
        case DType::F16: case DType::BF16: return 2;
        case DType::I64: return 8;
        case DType::I8: case DType::U8: case DType::BOOL: return 1;
        default: return 0;
    }
}

namespace {

DType parse_dtype(const std::string& s) {
    if (s == "F32")  return DType::F32;
    if (s == "F16")  return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "I64")  return DType::I64;
    if (s == "I32")  return DType::I32;
    if (s == "I8")   return DType::I8;
    if (s == "U8")   return DType::U8;
    if (s == "BOOL") return DType::BOOL;
    return DType::Unknown;
}

// bf16 is just the top 16 bits of an f32, so widening is a shift. This is the
// whole reason bf16 caught on for ML -- conversion is free.
inline float bf16_to_f32(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// IEEE half. Fiddlier than bf16 because the exponent is 5 bits instead of 8,
// so subnormals need renormalising by hand.
inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                      // signed zero
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

}  // namespace

int64_t TensorView::numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

std::string TensorView::shape_string() const {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i) o << ", ";
        o << shape[i];
    }
    o << "]";
    return o.str();
}

SafeTensors::SafeTensors(const std::string& path) {
    std::vector<std::string> files;

    if (fs::is_directory(path)) {
        for (const auto& e : fs::directory_iterator(path)) {
            if (e.path().extension() == ".safetensors") {
                files.push_back(e.path().string());
            }
        }
        // Shards are named model-00001-of-00003.safetensors, so sorting the
        // paths puts them in order. Not strictly required since every tensor
        // carries its own name, but it makes the load order predictable.
        std::sort(files.begin(), files.end());
        if (files.empty()) {
            throw std::runtime_error("no .safetensors files in " + path);
        }
    } else {
        files.push_back(path);
    }

    for (const auto& f : files) load_file(f);
}

void SafeTensors::load_file(const std::string& file) {
    int fd = ::open(file.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("can't open " + file);

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("fstat failed on " + file);
    }
    size_t fsize = static_cast<size_t>(st.st_size);
    if (fsize < 8) {
        ::close(fd);
        throw std::runtime_error("file too small to be safetensors: " + file);
    }

    void* base = ::mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("mmap failed on " + file);
    }

    maps_.push_back(Mapping{fd, static_cast<uint8_t*>(base), fsize});
    const uint8_t* p = static_cast<const uint8_t*>(base);

    uint64_t header_len = 0;
    std::memcpy(&header_len, p, 8);   // little-endian; every machine we target is LE

    if (header_len == 0 || 8 + header_len > fsize) {
        throw std::runtime_error("bogus header length in " + file);
    }

    std::string header_text(reinterpret_cast<const char*>(p + 8), header_len);
    json header = json::parse(header_text);

    const uint8_t* blob = p + 8 + header_len;
    size_t blob_size = fsize - 8 - header_len;

    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") continue;

        const json& t = it.value();
        TensorView v;
        v.name  = it.key();
        v.dtype = parse_dtype(t.at("dtype").get<std::string>());
        v.shape = t.at("shape").get<std::vector<int64_t>>();

        auto offsets = t.at("data_offsets").get<std::vector<uint64_t>>();
        if (offsets.size() != 2 || offsets[1] < offsets[0]) {
            throw std::runtime_error("bad data_offsets for tensor " + v.name);
        }
        uint64_t begin = offsets[0], end = offsets[1];
        if (end > blob_size) {
            throw std::runtime_error("tensor " + v.name + " runs past end of file");
        }

        v.data   = blob + begin;
        v.nbytes = static_cast<size_t>(end - begin);

        // Cheap sanity check that catches a misparsed header immediately
        // instead of 200 lines into the forward pass.
        size_t expect = static_cast<size_t>(v.numel()) * dtype_size(v.dtype);
        if (dtype_size(v.dtype) && expect != v.nbytes) {
            throw std::runtime_error("size mismatch for " + v.name +
                                     ": header says " + std::to_string(v.nbytes) +
                                     " bytes, shape implies " + std::to_string(expect));
        }

        tensors_.emplace(v.name, std::move(v));
    }
}

SafeTensors::~SafeTensors() {
    for (auto& m : maps_) {
        if (m.base) ::munmap(m.base, m.size);
        if (m.fd >= 0) ::close(m.fd);
    }
}

const TensorView* SafeTensors::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

bool SafeTensors::has(const std::string& name) const {
    return tensors_.count(name) > 0;
}

const TensorView& SafeTensors::get(const std::string& name) const {
    const TensorView* v = find(name);
    if (!v) throw std::runtime_error("no tensor named " + name);
    return *v;
}

std::vector<std::string> SafeTensors::names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& kv : tensors_) out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<float> SafeTensors::to_float(const std::string& name) const {
    const TensorView& v = get(name);
    size_t n = static_cast<size_t>(v.numel());
    std::vector<float> out(n);

    switch (v.dtype) {
        case DType::F32:
            std::memcpy(out.data(), v.data, n * sizeof(float));
            break;
        case DType::BF16: {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(v.data);
            for (size_t i = 0; i < n; i++) out[i] = bf16_to_f32(src[i]);
            break;
        }
        case DType::F16: {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(v.data);
            for (size_t i = 0; i < n; i++) out[i] = f16_to_f32(src[i]);
            break;
        }
        default:
            throw std::runtime_error("to_float doesn't handle dtype " +
                                     std::string(dtype_name(v.dtype)) +
                                     " (tensor " + name + ")");
    }
    return out;
}

}  // namespace verbum