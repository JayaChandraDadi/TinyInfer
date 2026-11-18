#include <gtest/gtest.h>

#include <random>
#include <tuple>
#include <vector>

#include "tinyinfer/gemm.h"

using namespace tinyinfer;

namespace {
std::vector<float> random_vec(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

void expect_close(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) ASSERT_NEAR(a[i], b[i], tol) << "at index " << i;
}
}  // namespace

class GemmShapes : public ::testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {};

TEST_P(GemmShapes, BlockedAndSimdMatchNaive) {
    auto [M, N, K] = GetParam();
    auto A = random_vec(static_cast<size_t>(M * K), 1);
    auto B = random_vec(static_cast<size_t>(K * N), 2);
    std::vector<float> ref(static_cast<size_t>(M * N)), blk(ref.size()), simd(ref.size());
    gemm_naive(M, N, K, A.data(), B.data(), ref.data(), false);
    gemm_blocked(M, N, K, A.data(), B.data(), blk.data(), false);
    gemm_simd(M, N, K, A.data(), B.data(), simd.data(), false);
    const float tol = 1e-4f * static_cast<float>(K);
    expect_close(ref, blk, tol);
    expect_close(ref, simd, tol);
}

// Odd sizes hit every edge path of the 4x16 micro-kernel and the cache blocks.
INSTANTIATE_TEST_SUITE_P(Sizes, GemmShapes,
                         ::testing::Values(std::make_tuple(1, 1, 1), std::make_tuple(3, 5, 7),
                                           std::make_tuple(4, 16, 8), std::make_tuple(5, 17, 3),
                                           std::make_tuple(16, 1024, 27), std::make_tuple(64, 100, 37),
                                           std::make_tuple(129, 257, 300), std::make_tuple(7, 9, 513)));

TEST(Gemm, AccumulateAddsIntoC) {
    const int64_t M = 9, N = 21, K = 13;
    auto A = random_vec(M * K, 3), B = random_vec(K * N, 4);
    std::vector<float> c0 = random_vec(M * N, 5), ref = c0, simd = c0;
    gemm_naive(M, N, K, A.data(), B.data(), ref.data(), true);
    gemm_simd(M, N, K, A.data(), B.data(), simd.data(), true);
    expect_close(ref, simd, 1e-4f);
}

TEST(Gemm, IdentityMatrix) {
    const int64_t n = 20;
    std::vector<float> I(n * n, 0.0f);
    for (int64_t i = 0; i < n; ++i) I[static_cast<size_t>(i * n + i)] = 1.0f;
    auto B = random_vec(n * n, 6);
    std::vector<float> C(n * n);
    gemm(n, n, n, I.data(), B.data(), C.data());
    expect_close(B, C, 1e-6f);
}

TEST(Gemm, ThreadCountDoesNotChangeResult) {
    const int64_t M = 70, N = 300, K = 90;
    auto A = random_vec(M * K, 7), B = random_vec(K * N, 8);
    std::vector<float> one(M * N), many(M * N);
    const int saved = get_num_threads();
    set_num_threads(1);
    gemm_simd(M, N, K, A.data(), B.data(), one.data(), false);
    set_num_threads(4);
    gemm_simd(M, N, K, A.data(), B.data(), many.data(), false);
    set_num_threads(saved);
    expect_close(one, many, 0.0f);  // same blocking => bitwise identical
}
