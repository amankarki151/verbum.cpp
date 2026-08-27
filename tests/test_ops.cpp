#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "verbum/ops.h"
#include "verbum/tensor.h"

using namespace verbum;

static int failures = 0;

static void check(bool cond, const std::string& what) {
    if (cond) {
        std::cout << "  ok    " << what << "\n";
    } else {
        std::cout << "  FAIL  " << what << "\n";
        failures++;
    }
}

static bool close(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(a) + std::fabs(b));
}

int main() {
    std::cout << "tensor basics\n";
    {
        Tensor t({2, 3});
        check(t.numel() == 6, "numel of [2,3] is 6");
        check(t.shape_string() == "[2, 3]", "shape prints as [2, 3]");
        check(t.data()[0] == 0.0f, "starts zeroed");

        t.at(1, 2) = 7.0f;
        check(t.data()[5] == 7.0f, "at(1,2) lands at flat index 5 (row major)");

        t.reshape({3, 2});
        check(t.at(2, 1) == 7.0f, "reshape keeps the data in place");

        bool threw = false;
        try { t.reshape({4, 4}); } catch (const std::exception&) { threw = true; }
        check(threw, "reshape to a different element count throws");
    }

    // [1 2 3]   [ 7  8 ]     [ 58  64 ]
    // [4 5 6] @ [ 9 10 ]  =  [139 154]
    //           [11 12]
    std::cout << "\nmatmul against a hand-computed example\n";
    {
        Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});
        Tensor c({2, 2});
        matmul(a, b, c);

        check(close(c.at(0, 0), 58.0f),  "c[0][0] == 58");
        check(close(c.at(0, 1), 64.0f),  "c[0][1] == 64");
        check(close(c.at(1, 0), 139.0f), "c[1][0] == 139");
        check(close(c.at(1, 1), 154.0f), "c[1][1] == 154");
    }

    std::cout << "\nmatmul_nt against the same example\n";
    {
        // B^T of the matrix above, so the answer must come out identical.
        Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor bt({2, 3}, {7, 9, 11, 8, 10, 12});
        Tensor c({2, 2});
        matmul_nt(a, bt, c);

        check(close(c.at(0, 0), 58.0f),  "c[0][0] == 58");
        check(close(c.at(0, 1), 64.0f),  "c[0][1] == 64");
        check(close(c.at(1, 0), 139.0f), "c[1][0] == 139");
        check(close(c.at(1, 1), 154.0f), "c[1][1] == 154");
    }

    std::cout << "\nidentity and shape errors\n";
    {
        Tensor a({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
        Tensor id({3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
        Tensor c({3, 3});
        matmul(a, id, c);
        bool same = true;
        for (int64_t i = 0; i < 9; i++) same &= close(c.data()[i], a.data()[i]);
        check(same, "A @ I == A");

        Tensor bad({4, 2});
        Tensor out({3, 2});
        bool threw = false;
        try { matmul(a, bad, out); } catch (const std::exception&) { threw = true; }
        check(threw, "mismatched inner dimensions throw");
    }

    // The two paths must agree on random data, not just on tidy integers.
    std::cout << "\nmatmul vs matmul_nt on random data\n";
    {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        const int64_t M = 17, K = 23, N = 13;
        Tensor a({M, K}), b({K, N}), bt({N, K});
        for (int64_t i = 0; i < a.numel(); i++) a.data()[i] = dist(rng);
        for (int64_t i = 0; i < b.numel(); i++) b.data()[i] = dist(rng);
        for (int64_t k = 0; k < K; k++)
            for (int64_t n = 0; n < N; n++)
                bt.at(n, k) = b.at(k, n);

        Tensor c1({M, N}), c2({M, N});
        matmul(a, b, c1);
        matmul_nt(a, bt, c2);

        bool same = true;
        for (int64_t i = 0; i < c1.numel(); i++) same &= close(c1.data()[i], c2.data()[i]);
        check(same, "both paths give the same answer");
    }

    // Baseline number. Day 7's CUDA matmul gets compared against this.
    std::cout << "\nbaseline throughput\n";
    {
        const int64_t M = 256, K = 1024, N = 1024;
        Tensor a({M, K}), b({N, K}), c({M, N});
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int64_t i = 0; i < a.numel(); i++) a.data()[i] = dist(rng);
        for (int64_t i = 0; i < b.numel(); i++) b.data()[i] = dist(rng);

        auto t0 = std::chrono::steady_clock::now();
        matmul_nt(a, b, c);
        auto t1 = std::chrono::steady_clock::now();

        double secs = std::chrono::duration<double>(t1 - t0).count();
        double gflop = 2.0 * M * K * N / 1e9;
        std::cout << "  " << M << "x" << K << " @ " << K << "x" << N
                  << "  " << secs * 1000 << " ms  "
                  << (gflop / secs) << " GFLOP/s\n";
    }

    std::cout << "\n" << (failures == 0 ? "all good" : "SOMETHING FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}