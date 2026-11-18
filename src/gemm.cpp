#include "tinyinfer/gemm.h"

#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define TINYINFER_AVX2 1
#endif

namespace tinyinfer {

EngineConfig& config() {
    static EngineConfig cfg;
    return cfg;
}

void set_num_threads(int n) {
    config().num_threads = n;
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#endif
}

int get_num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

bool simd_available() {
#ifdef TINYINFER_AVX2
    return true;
#else
    return false;
#endif
}

namespace {
void zero_or_keep(int64_t M, int64_t N, float* C, bool accumulate) {
    if (!accumulate) std::memset(C, 0, sizeof(float) * static_cast<size_t>(M * N));
}
}  // namespace

// Textbook i-j-k loop: strided access to B, no blocking, single thread.
void gemm_naive(int64_t M, int64_t N, int64_t K, const float* A, const float* B, float* C, bool accumulate) {
    for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j) {
            float acc = accumulate ? C[i * N + j] : 0.0f;
            for (int64_t k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

// Cache-blocked i-k-j loop, scalar. Auto-vectorisation is switched off so the
// benchmark isolates the gain from blocking/loop order vs. explicit SIMD.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
void gemm_blocked(int64_t M, int64_t N, int64_t K, const float* A, const float* B, float* C, bool accumulate) {
    constexpr int64_t KC = 256, NC = 256;
    zero_or_keep(M, N, C, accumulate);
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = 0; i < M; ++i)
        for (int64_t jb = 0; jb < N; jb += NC) {
            const int64_t je = std::min(jb + NC, N);
            for (int64_t kb = 0; kb < K; kb += KC) {
                const int64_t ke = std::min(kb + KC, K);
                float* c = C + i * N;
                for (int64_t k = kb; k < ke; ++k) {
                    const float a = A[i * K + k];
                    const float* b = B + k * N;
                    for (int64_t j = jb; j < je; ++j) c[j] += a * b[j];
                }
            }
        }
}

#ifdef TINYINFER_AVX2
namespace {
constexpr int64_t MR = 4;   // rows of C per micro-kernel
constexpr int64_t NR = 16;  // cols of C per micro-kernel (2 x 8-wide AVX registers)

// 4x16 register-blocked micro-kernel: 8 accumulators stay in YMM registers for
// the whole K panel; every B load is reused across 4 rows of A.
inline void kernel_4x16(int64_t N, int64_t K, const float* A, const float* B, float* C, int64_t kb, int64_t ke) {
    __m256 c00 = _mm256_loadu_ps(C + 0 * N), c01 = _mm256_loadu_ps(C + 0 * N + 8);
    __m256 c10 = _mm256_loadu_ps(C + 1 * N), c11 = _mm256_loadu_ps(C + 1 * N + 8);
    __m256 c20 = _mm256_loadu_ps(C + 2 * N), c21 = _mm256_loadu_ps(C + 2 * N + 8);
    __m256 c30 = _mm256_loadu_ps(C + 3 * N), c31 = _mm256_loadu_ps(C + 3 * N + 8);
    for (int64_t k = kb; k < ke; ++k) {
        const __m256 b0 = _mm256_loadu_ps(B + k * N);
        const __m256 b1 = _mm256_loadu_ps(B + k * N + 8);
        __m256 a = _mm256_broadcast_ss(A + 0 * K + k);
        c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(A + 1 * K + k);
        c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(A + 2 * K + k);
        c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(A + 3 * K + k);
        c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
    }
    _mm256_storeu_ps(C + 0 * N, c00); _mm256_storeu_ps(C + 0 * N + 8, c01);
    _mm256_storeu_ps(C + 1 * N, c10); _mm256_storeu_ps(C + 1 * N + 8, c11);
    _mm256_storeu_ps(C + 2 * N, c20); _mm256_storeu_ps(C + 2 * N + 8, c21);
    _mm256_storeu_ps(C + 3 * N, c30); _mm256_storeu_ps(C + 3 * N + 8, c31);
}

// Edge tiles (fewer than 4 rows or 16 cols): 8-wide vectors where possible, scalar tail.
inline void kernel_edge(int64_t mr, int64_t nr, int64_t N, int64_t K, const float* A, const float* B, float* C,
                        int64_t kb, int64_t ke) {
    for (int64_t r = 0; r < mr; ++r) {
        float* c = C + r * N;
        for (int64_t k = kb; k < ke; ++k) {
            const float a = A[r * K + k];
            const float* b = B + k * N;
            int64_t j = 0;
            const __m256 av = _mm256_set1_ps(a);
            for (; j + 8 <= nr; j += 8)
                _mm256_storeu_ps(c + j, _mm256_fmadd_ps(av, _mm256_loadu_ps(b + j), _mm256_loadu_ps(c + j)));
            for (; j < nr; ++j) c[j] += a * b[j];
        }
    }
}
}  // namespace

void gemm_simd(int64_t M, int64_t N, int64_t K, const float* A, const float* B, float* C, bool accumulate) {
    constexpr int64_t MC = 64, NC = 256, KC = 256;
    zero_or_keep(M, N, C, accumulate);
    const int64_t mblocks = (M + MC - 1) / MC, nblocks = (N + NC - 1) / NC;
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t mb = 0; mb < mblocks; ++mb)
        for (int64_t nb = 0; nb < nblocks; ++nb) {
            const int64_t i0 = mb * MC, i1 = std::min(i0 + MC, M);
            const int64_t j0 = nb * NC, j1 = std::min(j0 + NC, N);
            for (int64_t kb = 0; kb < K; kb += KC) {
                const int64_t ke = std::min(kb + KC, K);
                for (int64_t i = i0; i < i1; i += MR) {
                    const int64_t mr = std::min(MR, i1 - i);
                    for (int64_t j = j0; j < j1; j += NR) {
                        const int64_t nr = std::min(NR, j1 - j);
                        if (mr == MR && nr == NR)
                            kernel_4x16(N, K, A + i * K, B + j, C + i * N + j, kb, ke);
                        else
                            kernel_edge(mr, nr, N, K, A + i * K, B + j, C + i * N + j, kb, ke);
                    }
                }
            }
        }
}
#else
void gemm_simd(int64_t M, int64_t N, int64_t K, const float* A, const float* B, float* C, bool accumulate) {
    gemm_blocked(M, N, K, A, B, C, accumulate);
}
#endif

void gemm(int64_t M, int64_t N, int64_t K, const float* A, const float* B, float* C, bool accumulate,
          GemmKernel kernel) {
    switch (kernel) {
        case GemmKernel::Naive: gemm_naive(M, N, K, A, B, C, accumulate); break;
        case GemmKernel::Blocked: gemm_blocked(M, N, K, A, B, C, accumulate); break;
        case GemmKernel::Simd: gemm_simd(M, N, K, A, B, C, accumulate); break;
    }
}

}  // namespace tinyinfer
