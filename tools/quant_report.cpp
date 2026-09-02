#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "verbum/quant.h"
#include "verbum/safetensors.h"
#include "verbum/tensor.h"

using namespace verbum;

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "models/qwen3-0.6b";

    SafeTensors st(dir);

    size_t f32_total = 0, q8_total = 0;
    double worst_err = 0.0;
    std::string worst_name;

    printf("%-52s %10s %10s\n", "tensor", "params", "rel RMS");

    for (const auto& name : st.names()) {
        const TensorView& v = st.get(name);
        if (v.shape.size() != 2) continue;              // skip norms (1-D)
        if (name.find("embed_tokens") != std::string::npos) continue;
        if (name.find("lm_head") != std::string::npos) continue;

        Tensor w(v.shape, st.to_float(name));
        QuantTensor q = quantize_rows(w);
        Tensor back = dequantize(q);

        double sse = 0, sr = 0;
        for (int64_t i = 0; i < w.numel(); i++) {
            const double e = (double)w.data()[i] - back.data()[i];
            sse += e * e;
            sr += (double)w.data()[i] * w.data()[i];
        }
        const double err = std::sqrt(sse / sr);

        f32_total += (size_t)w.numel() * sizeof(float);
        q8_total += q.bytes();

        if (err > worst_err) { worst_err = err; worst_name = name; }

        // print only the first layer's tensors, otherwise it's 200 lines
        if (name.find("layers.0.") != std::string::npos) {
            printf("%-52s %10lld %9.4f%%\n", name.c_str(),
                   (long long)w.numel(), err * 100);
        }
    }

    printf("\nacross all quantized weight matrices:\n");
    printf("  f32:  %.2f MB\n", f32_total / 1e6);
    printf("  int8: %.2f MB\n", q8_total / 1e6);
    printf("  %.2fx smaller\n", (double)f32_total / q8_total);
    printf("  worst tensor: %s at %.4f%%\n", worst_name.c_str(), worst_err * 100);
    return 0;
}