#pragma once

#include <map>
#include <string>
#include <vector>

#include "tinyinfer/tensor.h"

namespace tinyinfer {

// A small, dependency-free reader for the subset of the ONNX protobuf schema that
// an inference engine needs: graph nodes, attributes, initializers and I/O names.
// It decodes the protobuf wire format directly, so no protobuf/onnx libraries are
// linked into the engine.

struct Attribute {
    enum class Kind { None, Float, Int, String, Tensor, Floats, Ints };
    Kind kind = Kind::None;
    float f = 0.0f;
    int64_t i = 0;
    std::string s;
    Tensor t;
    std::vector<float> floats;
    std::vector<int64_t> ints;
};

struct Node {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::map<std::string, Attribute> attrs;

    bool has(const std::string& k) const { return attrs.count(k) > 0; }
    int64_t get_int(const std::string& k, int64_t def) const;
    float get_float(const std::string& k, float def) const;
    std::string get_string(const std::string& k, const std::string& def) const;
    std::vector<int64_t> get_ints(const std::string& k, std::vector<int64_t> def) const;
};

struct ValueInfo {
    std::string name;
    Shape shape;  // -1 for symbolic dims
};

struct ModelProto {
    int64_t ir_version = 0;
    int64_t opset = 0;
    std::string producer;
    std::vector<Node> nodes;
    std::map<std::string, Tensor> initializers;
    std::vector<ValueInfo> inputs;   // graph inputs, excluding initializers
    std::vector<ValueInfo> outputs;
};

ModelProto load_onnx(const std::string& path);
ModelProto parse_onnx(const std::string& bytes);

}  // namespace tinyinfer
