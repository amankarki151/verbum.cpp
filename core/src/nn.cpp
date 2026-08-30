#include "verbum/nn.h"

#include <cmath>
#include <stdexcept>

namespace verbum {

void rmsnorm(const Tensor& x, const std::vector<float>& weight, float eps,
             Tensor& out) {
    if (x.rank() != 2 || out.rank() != 2) {
        throw std::runtime_error("rmsnorm wants 2-D tensors");
    }
    const int64_t rows = x.dim(0), dim = x.dim(1);
    if (out.dim(0) != rows || out.dim(1) != dim) {
        throw std::runtime_error("rmsnorm output is the wrong shape");
    }
    if (static_cast<int64_t>(weight.size()) != dim) {
        throw std::runtime_error("rmsnorm weight length doesn't match dim");
    }

    const float* X = x.data();
    float* O = out.data();

    for (int64_t r = 0; r < rows; r++) {
        const float* xr = X + r * dim;
        float* orow = O + r * dim;

        // Accumulate in double. The sum runs over the full hidden size and
        // gets divided down, so float accumulation drifts enough here to show
        // up when comparing against the reference later.
        double sumsq = 0.0;
        for (int64_t i = 0; i < dim; i++) {
            sumsq += static_cast<double>(xr[i]) * static_cast<double>(xr[i]);
        }
        const float scale =
            1.0f / std::sqrt(static_cast<float>(sumsq / static_cast<double>(dim)) + eps);

        for (int64_t i = 0; i < dim; i++) {
            orow[i] = xr[i] * scale * weight[i];
        }
    }
}

RopeTable build_rope_table(int head_dim, int max_pos, float theta) {
    if (head_dim % 2 != 0) throw std::runtime_error("head_dim must be even for rope");

    RopeTable t;
    t.head_dim = head_dim;
    t.max_pos = max_pos;
    t.cos.resize(static_cast<size_t>(max_pos) * head_dim);
    t.sin.resize(static_cast<size_t>(max_pos) * head_dim);

    const int half = head_dim / 2;

    for (int pos = 0; pos < max_pos; pos++) {
        for (int i = 0; i < half; i++) {
            // inv_freq[i] = 1 / theta^(2i/head_dim)
            const double inv_freq =
                1.0 / std::pow(static_cast<double>(theta),
                               static_cast<double>(2 * i) / static_cast<double>(head_dim));
            const double angle = static_cast<double>(pos) * inv_freq;
            const float c = static_cast<float>(std::cos(angle));
            const float s = static_cast<float>(std::sin(angle));

            // The table is duplicated across both halves, matching how HF
            // concatenates freqs with itself before taking cos/sin.
            t.cos[static_cast<size_t>(pos) * head_dim + i] = c;
            t.cos[static_cast<size_t>(pos) * head_dim + i + half] = c;
            t.sin[static_cast<size_t>(pos) * head_dim + i] = s;
            t.sin[static_cast<size_t>(pos) * head_dim + i + half] = s;
        }
    }
    return t;
}

void apply_rope(Tensor& x, int pos, const RopeTable& table) {
    if (x.rank() != 2) throw std::runtime_error("apply_rope wants a 2-D tensor");
    const int64_t heads = x.dim(0), dim = x.dim(1);
    if (dim != table.head_dim) {
        throw std::runtime_error("head_dim doesn't match the rope table");
    }
    if (pos < 0 || pos >= table.max_pos) {
        throw std::runtime_error("position is outside the rope table");
    }

    const int half = table.head_dim / 2;
    const float* cos = table.cos.data() + static_cast<size_t>(pos) * table.head_dim;
    const float* sin = table.sin.data() + static_cast<size_t>(pos) * table.head_dim;

    for (int64_t h = 0; h < heads; h++) {
        float* row = x.data() + h * dim;
        // Read both halves before writing either -- rotating in place while
        // reading is the classic way to corrupt half the vector.
        for (int i = 0; i < half; i++) {
            const float a = row[i];
            const float b = row[i + half];
            row[i]        = a * cos[i]        - b * sin[i];
            row[i + half] = b * cos[i + half] + a * sin[i + half];
        }
    }
}

void embedding_lookup(const std::vector<float>& embed_matrix, int64_t dim,
                      const std::vector<int>& ids, Tensor& out) {
    const int64_t n = static_cast<int64_t>(ids.size());
    if (out.rank() != 2 || out.dim(0) != n || out.dim(1) != dim) {
        throw std::runtime_error("embedding output is the wrong shape");
    }
    const int64_t vocab = static_cast<int64_t>(embed_matrix.size()) / dim;

    for (int64_t i = 0; i < n; i++) {
        const int id = ids[static_cast<size_t>(i)];
        if (id < 0 || id >= vocab) {
            throw std::runtime_error("token id out of range for the embedding table");
        }
        const float* src = embed_matrix.data() + static_cast<int64_t>(id) * dim;
        float* dst = out.data() + i * dim;
        for (int64_t k = 0; k < dim; k++) dst[k] = src[k];
    }
}

}  // namespace verbum