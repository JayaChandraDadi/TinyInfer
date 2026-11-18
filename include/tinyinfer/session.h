#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tinyinfer/onnx_reader.h"
#include "tinyinfer/tensor.h"

namespace tinyinfer {

struct OpProfile {
    std::string op_type;
    int64_t calls = 0;
    double total_ms = 0.0;
};

// Loads an ONNX model once and runs it many times. Nodes execute in the graph's
// topological order; intermediate tensors are released after their last use.
class Session {
public:
    explicit Session(const std::string& onnx_path);
    explicit Session(ModelProto model);

    // Single-input convenience overload.
    std::vector<Tensor> run(const Tensor& input);
    std::vector<Tensor> run(const std::map<std::string, Tensor>& feeds);

    const std::vector<std::string>& input_names() const { return input_names_; }
    const std::vector<std::string>& output_names() const { return output_names_; }
    const std::vector<Shape>& input_shapes() const { return input_shapes_; }
    std::vector<std::string> op_types() const;
    size_t num_nodes() const { return model_.nodes.size(); }

    void enable_profiling(bool on) { profiling_ = on; }
    std::vector<OpProfile> profile() const;
    void reset_profile() { profile_.clear(); }

    static std::vector<std::string> supported_ops();

private:
    void prepare();

    ModelProto model_;
    std::vector<std::string> input_names_, output_names_;
    std::vector<Shape> input_shapes_;
    std::unordered_map<std::string, Tensor> constants_;      // initializers + Constant nodes
    std::vector<std::vector<std::string>> release_after_;    // per node: values that die after it
    bool profiling_ = false;
    std::map<std::string, OpProfile> profile_;
};

}  // namespace tinyinfer
