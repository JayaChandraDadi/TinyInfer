// ti_bench: latency of one model under each kernel configuration.
//   ti_bench model.onnx [--batch N] [--iters K] [--warmup W] [--threads T] [--skip-naive]
// Prints a markdown table (median and p90 latency per forward pass).

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "tinyinfer/gemm.h"
#include "tinyinfer/session.h"

using namespace tinyinfer;

namespace {
struct Variant {
    std::string name;
    ConvAlgo conv;
    GemmKernel gemm;
    int threads;
    bool slow;
};

double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const size_t i = std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size() - 1) + 0.5));
    return v[i];
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: ti_bench model.onnx [--batch N] [--iters K] [--warmup W] [--threads T] [--skip-naive]\n";
        return 2;
    }
    int64_t batch = 1;
    int iters = 50, warmup = 5, max_threads = get_num_threads();
    bool skip_naive = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--batch") batch = std::stoll(argv[++i]);
        else if (a == "--iters") iters = std::stoi(argv[++i]);
        else if (a == "--warmup") warmup = std::stoi(argv[++i]);
        else if (a == "--threads") max_threads = std::stoi(argv[++i]);
        else if (a == "--skip-naive") skip_naive = true;
    }

    try {
    Session s(argv[1]);
    Shape shape = s.input_shapes()[0];
    for (auto& d : shape) if (d < 0) d = 1;
    shape[0] = batch;
    Tensor x(shape);
    std::mt19937 rng(0);
    std::normal_distribution<float> nd;
    for (auto& v : x.data) v = nd(rng);

    const std::vector<Variant> variants = {
        {"direct conv (7 nested loops)", ConvAlgo::Direct, GemmKernel::Naive, 1, true},
        {"im2col + naive GEMM", ConvAlgo::Im2colGemm, GemmKernel::Naive, 1, true},
        {"im2col + blocked GEMM (scalar)", ConvAlgo::Im2colGemm, GemmKernel::Blocked, 1, false},
        {"im2col + AVX2/FMA GEMM", ConvAlgo::Im2colGemm, GemmKernel::Simd, 1, false},
        {"im2col + AVX2/FMA GEMM + OpenMP", ConvAlgo::Im2colGemm, GemmKernel::Simd, max_threads, false},
    };

    std::cout << "model: " << argv[1] << "  input: " << shape_str(shape) << "  simd: " << (simd_available() ? "AVX2+FMA" : "off")
              << "\n\n| configuration | threads | median ms | p90 ms | speed-up |\n|---|---:|---:|---:|---:|\n";
    double baseline = 0.0;
    for (const auto& v : variants) {
        if (v.slow && skip_naive) continue;
        config().conv_algo = v.conv;
        config().gemm_kernel = v.gemm;
        set_num_threads(v.threads);
        const int n = v.slow ? std::max(3, iters / 10) : iters;
        for (int i = 0; i < (v.slow ? 1 : warmup); ++i) s.run(x);
        std::vector<double> ms;
        for (int i = 0; i < n; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            s.run(x);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const double med = percentile(ms, 0.5);
        if (baseline == 0.0) baseline = med;
        std::cout << "| " << v.name << " | " << v.threads << " | " << std::fixed << std::setprecision(2) << med << " | "
                  << percentile(ms, 0.9) << " | " << std::setprecision(1) << baseline / med << "x |\n";
    }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
