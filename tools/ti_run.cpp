// ti_run: load an ONNX model, run it once and print the output summary.
//   ti_run model.onnx [input.bin] [--profile]

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>

#include "tinyinfer/session.h"
#include "tinyinfer/tensor_io.h"

using namespace tinyinfer;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: ti_run model.onnx [input.bin] [--profile]\n";
        return 2;
    }
    bool profile = false;
    std::string input_path;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--profile") == 0) profile = true;
        else input_path = argv[i];
    }
    try {
        Session s(argv[1]);
        std::cout << "model: " << argv[1] << "  nodes: " << s.num_nodes() << "\nops:";
        for (auto& op : s.op_types()) std::cout << ' ' << op;
        std::cout << "\ninput: " << s.input_names()[0] << ' ' << shape_str(s.input_shapes()[0]) << '\n';

        Tensor x;
        if (!input_path.empty()) {
            x = read_tensor(input_path);
        } else {
            Shape shape = s.input_shapes()[0];
            for (auto& d : shape) if (d < 0) d = 1;
            x = Tensor(shape);
            std::mt19937 rng(0);
            std::normal_distribution<float> nd;
            for (auto& v : x.data) v = nd(rng);
        }
        s.enable_profiling(profile);
        auto outs = s.run(x);
        for (size_t i = 0; i < outs.size(); ++i) {
            const auto& o = outs[i];
            std::cout << "output " << s.output_names()[i] << ' ' << shape_str(o.shape) << " first values:";
            for (int64_t j = 0; j < std::min<int64_t>(o.size(), 10); ++j)
                std::cout << ' ' << std::setprecision(5) << o.data[static_cast<size_t>(j)];
            std::cout << '\n';
        }
        if (profile) {
            std::cout << "\nper-op time:\n";
            for (auto& p : s.profile())
                std::cout << "  " << std::setw(20) << std::left << p.op_type << std::setw(6) << p.calls
                          << std::fixed << std::setprecision(3) << p.total_ms << " ms\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
