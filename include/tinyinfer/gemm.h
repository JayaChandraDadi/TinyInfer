#pragma once

#include <cstdint>

namespace tinyinfer {

// Kernel selection. The defaults are the fast path; the others exist so the
// benchmark can show what each optimisation buys.
enum class ConvAlgo { Direct, Im2colGemm };
enum class GemmKernel { Naive, Blocked, Simd };

struct EngineConfig {
    ConvAlgo conv_algo = ConvAlgo::Im2colGemm;
    GemmKernel gemm_kernel = GemmKernel::Simd;
    int num_threads = 0;  // 0 = OpenMP default
};

EngineConfig& config();
void set_num_threads(int n);
int get_num_threads();
bool simd_available();  // true when compiled with AVX2+FMA

// C[M,N] = A[M,K] * B[K,N] (+ C if accumulate). All row-major, contiguous,
// leading dimensions equal to the logical widths.
void gemm(int64_t M, int64_t N, int64_t K, const float* A, const float* B, float* C,
          bool accumulate = false, GemmKernel kernel = config().gemm_kernel);

void gemm_naive(int64_t M, int64_t N, int64_t K, const float* A, const float* B, float* C, bool accumulate);
void gemm_blocked(int64_t M, int64_t N, int64_t K, const float* A, const float* B, float* C, bool accumulate);
void gemm_simd(int64_t M, int64_t N, int64_t K, const float* A, const float* B, float* C, bool accumulate);

}  // namespace tinyinfer
