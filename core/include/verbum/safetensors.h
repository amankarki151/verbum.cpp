#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace verbum {

enum class DType { F32, F16, BF16, I64, I32, I8, U8, BOOL, Unknown };

const char* dtype_name(DType d);
size_t dtype_size(DType d);

// A window onto tensor bytes inside the mmap'd file. Doesn't own anything --
// it stays valid as long as the SafeTensors object that produced it is alive.
struct TensorView {
    std::string name;
    DType dtype = DType::Unknown;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;

    int64_t numel() const;
    std::string shape_string() const;
};

class SafeTensors {
public:
    // Takes a single .safetensors file, or a folder containing one or more shards.
    explicit SafeTensors(const std::string& path);
    ~SafeTensors();

    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;

    const TensorView& get(const std::string& name) const;  // throws if absent
    const TensorView* find(const std::string& name) const; // null if absent
    bool has(const std::string& name) const;

    std::vector<std::string> names() const;
    size_t size() const { return tensors_.size(); }

    // Materialise a tensor as float32, converting from bf16/f16 if needed.
    // Allocates. Use it for tests and small tensors, not the whole model.
    std::vector<float> to_float(const std::string& name) const;

private:
    struct Mapping {
        int fd = -1;
        uint8_t* base = nullptr;
        size_t size = 0;
    };

    void load_file(const std::string& file);

    std::vector<Mapping> maps_;
    std::unordered_map<std::string, TensorView> tensors_;
};

}  // namespace verbum