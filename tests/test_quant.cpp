#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "verbum/ops.h"
#include "verbum/quant.h"
#include "verbum/tensor.h"

using namespace verbum;

static int failures = 0;

static void check(bool cond, const std::string& what) {
    if (cond) std::cout << "  ok    " << what << "\n";
    else { std::cout << "  FAIL  " << what << "\n"; failures++; }
}

// Relative RMS error -- the right measure for quantization, since absolute
// error scales with the magnitude of the values.
static double rel_rms(const float* ref, const float* got, size_t n) {
    double sse = 0, sr = 0;
    for (size_t i = 0; i < n; i++) {
        const double e = (double)ref[i] - got[i];
        sse += e * e;
        sr += (double)ref[i] * ref[i];
    }
    return (sr > 0) ? std::sqrt(sse / sr) : 0.0;
}

int main() {
    std::mt19937 rng(11);
    // Transformer weights are roughly normal with small std -- use that, not
    // uniform, so the measured error is representative of the real thing.
    std::normal_distribution<float> wd(0.f, 0.02f);
    std::normal_distribution<float> ad(0.f, 1.0f);

    std::cout << "exact cases\n";
    {
        // A row whose values are exactly representable should round-trip
        // with zero error.
        Tensor w({1, 3}, {127.0f, -127.0f, 0.0f});
        QuantTensor q = quantize_rows(w);
        Tensor back = dequantize(q);
        check(back.at(0,0) == 127.0f && back.at(0,1) == -127.0f && back.at(0,2) == 0.0f,
              "exactly-representable row round-trips exactly");

        // Zero maps to zero -- that's what symmetric quantization buys.
        Tensor z({1, 4}, {0.f, 0.f, 0.f, 0.f});
        QuantTensor qz = quantize_rows(z);
        Tensor bz = dequantize(qz);
        bool all_zero = true;
        for (int i = 0; i < 4; i++) all_zero &= (bz.data()[i] == 0.0f);
        check(all_zero, "an all-zero row stays all zero");

        // Every quantized value must fit the symmetric range.
        Tensor big({2, 5}, {1e6f, -1e6f, 0.f, 5.f, -5.f, 1.f, 2.f, 3.f, 4.f, 5.f});
        QuantTensor qb = quantize_rows(big);
        bool in_range = true;
        for (int8_t v : qb.data) in_range &= (v >= -127 && v <= 127);
        check(in_range, "all values land inside [-127, 127]");
    }

    std::cout << "\nround-trip error on realistic weights\n";
    for (int64_t cols : {1024, 3072}) {
        const int64_t rows = 256;
        Tensor w({rows, cols});
        for (int64_t i = 0; i < w.numel(); i++) w.data()[i] = wd(rng);

        QuantTensor q = quantize_rows(w);
        Tensor back = dequantize(q);
        const double err = rel_rms(w.data(), back.data(), (size_t)w.numel());

        check(err < 0.02,
              "cols=" + std::to_string(cols) + " rel RMS error " +
                  std::to_string(err * 100) + "%");
    }

    std::cout << "\nmatmul output error, f32 weights vs int8 weights\n";
    for (auto dims : std::vector<std::array<int64_t,3>>{
             {1, 1024, 1024}, {1, 3072, 1024}, {8, 1024, 1024}}) {
        const int64_t M = dims[0], N = dims[1], K = dims[2];
        Tensor A({M, K}), B({N, K}), C1({M, N}), C2({M, N});
        for (int64_t i = 0; i < A.numel(); i++) A.data()[i] = ad(rng);
        for (int64_t i = 0; i < B.numel(); i++) B.data()[i] = wd(rng);

        matmul_nt(A, B, C1);
        QuantTensor q = quantize_rows(B);
        matmul_nt_q8(A, q, C2);

        const double err = rel_rms(C1.data(), C2.data(), (size_t)C1.numel());
        check(err < 0.02,
              "M=" + std::to_string(M) + " N=" + std::to_string(N) +
                  " rel RMS " + std::to_string(err * 100) + "%");
    }

    std::cout << "\nper-row scales vs one global scale\n";
    {
        // Deliberately give half the rows a 50x smaller magnitude, which is
        // the situation per-row scaling exists to handle.
        const int64_t rows = 64, cols = 1024;
        Tensor w({rows, cols});
        for (int64_t r = 0; r < rows; r++) {
            std::normal_distribution<float> d(0.f, (r % 2 == 0) ? 0.02f : 0.0004f);
            for (int64_t c = 0; c < cols; c++) w.at(r, c) = d(rng);
        }

        QuantTensor q = quantize_rows(w);
        Tensor back = dequantize(q);
        const double per_row = rel_rms(w.data(), back.data(), (size_t)w.numel());

        float amax = 0;
        for (int64_t i = 0; i < w.numel(); i++) amax = std::max(amax, std::fabs(w.data()[i]));
        const float g = amax / 127.0f;
        std::vector<float> global((size_t)w.numel());
        for (int64_t i = 0; i < w.numel(); i++) {
            int v = std::max(-127, std::min(127, (int)std::lrintf(w.data()[i] / g)));
            global[(size_t)i] = v * g;
        }
        const double global_err = rel_rms(w.data(), global.data(), (size_t)w.numel());

        std::cout << "    per-row:  " << per_row * 100 << "%\n";
        std::cout << "    global:   " << global_err * 100 << "%\n";
        check(per_row < global_err, "per-row scales beat a single global scale");
    }

    std::cout << "\nmemory\n";
    {
        Tensor w({1000, 1000});
        for (int64_t i = 0; i < w.numel(); i++) w.data()[i] = wd(rng);
        QuantTensor q = quantize_rows(w);
        const size_t f32_bytes = (size_t)w.numel() * sizeof(float);
        const double ratio = (double)f32_bytes / q.bytes();
        std::cout << "    f32: " << f32_bytes / 1024 << " KB, int8: "
                  << q.bytes() / 1024 << " KB, ratio " << ratio << "x\n";
        check(ratio > 3.9, "roughly 4x smaller");
    }

    std::cout << "\n" << (failures == 0 ? "all good" : "SOMETHING FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}