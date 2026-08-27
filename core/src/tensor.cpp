#include "verbum/tensor.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace verbum {

namespace {
int64_t product(const std::vector<int64_t>& s) {
    int64_t n = 1;
    for (int64_t d : s) {
        if (d < 0) throw std::runtime_error("negative dimension");
        n *= d;
    }
    return n;
}
}  // namespace

Tensor::Tensor(std::vector<int64_t> shape) : shape_(std::move(shape)) {
    numel_ = product(shape_);
    data_.assign(static_cast<size_t>(numel_), 0.0f);
}

Tensor::Tensor(std::vector<int64_t> shape, std::vector<float> data)
    : shape_(std::move(shape)), data_(std::move(data)) {
    numel_ = product(shape_);
    if (static_cast<size_t>(numel_) != data_.size()) {
        throw std::runtime_error("shape doesn't match data size");
    }
}

void Tensor::reshape(std::vector<int64_t> shape) {
    int64_t n = product(shape);
    if (n != numel_) throw std::runtime_error("reshape would change element count");
    shape_ = std::move(shape);
}

void Tensor::fill(float v) {
    for (auto& x : data_) x = v;
}

float& Tensor::at(int64_t r, int64_t c) {
    assert(rank() == 2 && r >= 0 && r < shape_[0] && c >= 0 && c < shape_[1]);
    return data_[static_cast<size_t>(r * shape_[1] + c)];
}

const float& Tensor::at(int64_t r, int64_t c) const {
    assert(rank() == 2 && r >= 0 && r < shape_[0] && c >= 0 && c < shape_[1]);
    return data_[static_cast<size_t>(r * shape_[1] + c)];
}

std::string Tensor::shape_string() const {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < shape_.size(); i++) {
        if (i) o << ", ";
        o << shape_[i];
    }
    o << "]";
    return o.str();
}

}  // namespace verbum