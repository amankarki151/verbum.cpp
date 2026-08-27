#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace verbum {

// A dense, contiguous, row-major float32 tensor that owns its memory.
// Deliberately plain: no broadcasting, no views, no lazy anything. Every
// buffer the forward pass needs gets allocated once and reused.
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<int64_t> shape);
    Tensor(std::vector<int64_t> shape, std::vector<float> data);

    void reshape(std::vector<int64_t> shape);
    void fill(float v);
    void zero() { fill(0.0f); }

    float*       data()       { return data_.data(); }
    const float* data() const { return data_.data(); }

    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t dim(size_t i) const { return shape_[i]; }
    size_t  rank() const { return shape_.size(); }
    int64_t numel() const { return numel_; }

    // 2-D access. Bounds-checked in debug builds only.
    float&       at(int64_t r, int64_t c);
    const float& at(int64_t r, int64_t c) const;

    std::string shape_string() const;

private:
    std::vector<int64_t> shape_;
    std::vector<float> data_;
    int64_t numel_ = 0;
};

}  // namespace verbum