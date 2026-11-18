#include "tinyinfer/session.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>

#include "tinyinfer/ops.h"

namespace tinyinfer {

namespace {

using Inputs = std::vector<const Tensor*>;
using OpFn = std::function<Tensor(const Node&, const Inputs&)>;

const Tensor& in(const Inputs& xs, size_t i, const Node& n) {
    if (i >= xs.size() || !xs[i]) throw std::runtime_error(n.op_type + " '" + n.name + "': missing input " + std::to_string(i));
    return *xs[i];
}
const Tensor* opt(const Inputs& xs, size_t i) { return i < xs.size() ? xs[i] : nullptr; }

std::array<int64_t, 2> pair_attr(const Node& n, const std::string& k, int64_t def) {
    auto v = n.get_ints(k, {def, def});
    if (v.size() == 1) v.push_back(v[0]);
    return {v[0], v[1]};
}

std::array<int64_t, 4> pads_attr(const Node& n) {
    auto v = n.get_ints("pads", {0, 0, 0, 0});
    if (v.size() != 4) throw std::runtime_error(n.op_type + ": only 2-D pads are supported");
    const std::string ap = n.get_string("auto_pad", "NOTSET");
    if (ap != "NOTSET" && ap != "VALID") throw std::runtime_error(n.op_type + ": auto_pad=" + ap + " is not supported");
    return {v[0], v[1], v[2], v[3]};
}

const std::map<std::string, OpFn>& registry() {
    static const std::map<std::string, OpFn> ops = {
        {"Conv", [](const Node& n, const Inputs& x) {
             ops::Conv2dParams p;
             p.stride = pair_attr(n, "strides", 1);
             p.dilation = pair_attr(n, "dilations", 1);
             p.pads = pads_attr(n);
             p.group = n.get_int("group", 1);
             return ops::conv2d(in(x, 0, n), in(x, 1, n), opt(x, 2), p);
         }},
        {"MaxPool", [](const Node& n, const Inputs& x) {
             ops::Pool2dParams p;
             p.kernel = pair_attr(n, "kernel_shape", 1);
             p.stride = pair_attr(n, "strides", 1);
             p.pads = pads_attr(n);
             p.ceil_mode = n.get_int("ceil_mode", 0) != 0;
             if (n.get_ints("dilations", {1, 1}) != std::vector<int64_t>{1, 1})
                 throw std::runtime_error("MaxPool: dilations are not supported");
             return ops::maxpool2d(in(x, 0, n), p);
         }},
        {"AveragePool", [](const Node& n, const Inputs& x) {
             ops::Pool2dParams p;
             p.kernel = pair_attr(n, "kernel_shape", 1);
             p.stride = pair_attr(n, "strides", 1);
             p.pads = pads_attr(n);
             p.ceil_mode = n.get_int("ceil_mode", 0) != 0;
             p.count_include_pad = n.get_int("count_include_pad", 0) != 0;
             return ops::avgpool2d(in(x, 0, n), p);
         }},
        {"GlobalAveragePool", [](const Node& n, const Inputs& x) { return ops::global_avgpool(in(x, 0, n)); }},
        {"Relu", [](const Node& n, const Inputs& x) { return ops::relu(in(x, 0, n)); }},
        {"Sigmoid", [](const Node& n, const Inputs& x) { return ops::sigmoid(in(x, 0, n)); }},
        {"HardSigmoid", [](const Node& n, const Inputs& x) {
             return ops::hardsigmoid(in(x, 0, n), n.get_float("alpha", 0.2f), n.get_float("beta", 0.5f));
         }},
        {"HardSwish", [](const Node& n, const Inputs& x) { return ops::hardswish(in(x, 0, n)); }},
        {"Clip", [](const Node& n, const Inputs& x) {
             float lo = -std::numeric_limits<float>::infinity(), hi = std::numeric_limits<float>::infinity();
             if (auto* t = opt(x, 1); t && t->size()) lo = t->data[0];
             if (auto* t = opt(x, 2); t && t->size()) hi = t->data[0];
             if (n.has("min")) lo = n.get_float("min", lo);  // opset < 11
             if (n.has("max")) hi = n.get_float("max", hi);
             return ops::clip(in(x, 0, n), lo, hi);
         }},
        {"BatchNormalization", [](const Node& n, const Inputs& x) {
             return ops::batchnorm(in(x, 0, n), in(x, 1, n), in(x, 2, n), in(x, 3, n), in(x, 4, n),
                                   n.get_float("epsilon", 1e-5f));
         }},
        {"Gemm", [](const Node& n, const Inputs& x) {
             return ops::gemm(in(x, 0, n), in(x, 1, n), opt(x, 2), n.get_int("transA", 0) != 0,
                              n.get_int("transB", 0) != 0, n.get_float("alpha", 1.0f), n.get_float("beta", 1.0f));
         }},
        {"MatMul", [](const Node& n, const Inputs& x) { return ops::matmul(in(x, 0, n), in(x, 1, n)); }},
        {"Add", [](const Node& n, const Inputs& x) { return ops::binary(in(x, 0, n), in(x, 1, n), ops::BinOp::Add); }},
        {"Sub", [](const Node& n, const Inputs& x) { return ops::binary(in(x, 0, n), in(x, 1, n), ops::BinOp::Sub); }},
        {"Mul", [](const Node& n, const Inputs& x) { return ops::binary(in(x, 0, n), in(x, 1, n), ops::BinOp::Mul); }},
        {"Div", [](const Node& n, const Inputs& x) { return ops::binary(in(x, 0, n), in(x, 1, n), ops::BinOp::Div); }},
        {"Flatten", [](const Node& n, const Inputs& x) { return ops::flatten(in(x, 0, n), n.get_int("axis", 1)); }},
        {"Reshape", [](const Node& n, const Inputs& x) {
             const Tensor& s = in(x, 1, n);
             if (!s.is_int) throw std::runtime_error("Reshape: shape input must be int64");
             return ops::reshape(in(x, 0, n), s.i64, n.get_int("allowzero", 0) != 0);
         }},
        {"Softmax", [](const Node& n, const Inputs& x) { return ops::softmax(in(x, 0, n), n.get_int("axis", -1)); }},
        {"Identity", [](const Node& n, const Inputs& x) { return in(x, 0, n); }},
        {"Dropout", [](const Node& n, const Inputs& x) { return in(x, 0, n); }},  // inference: no-op
    };
    return ops;
}

}  // namespace

std::vector<std::string> Session::supported_ops() {
    std::vector<std::string> v;
    for (auto& kv : registry()) v.push_back(kv.first);
    v.push_back("Constant");
    std::sort(v.begin(), v.end());
    return v;
}

Session::Session(const std::string& onnx_path) : Session(load_onnx(onnx_path)) {}

Session::Session(ModelProto model) : model_(std::move(model)) { prepare(); }

void Session::prepare() {
    for (auto& kv : model_.initializers) constants_[kv.first] = kv.second;
    model_.initializers.clear();

    // Fold Constant nodes and validate every op up front, so unsupported models
    // fail at load time rather than halfway through the first request.
    std::vector<Node> runnable;
    for (auto& n : model_.nodes) {
        if (n.op_type == "Constant") {
            auto it = n.attrs.find("value");
            if (it == n.attrs.end() || it->second.kind != Attribute::Kind::Tensor)
                throw std::runtime_error("Constant: only tensor 'value' attributes are supported");
            constants_[n.outputs.at(0)] = it->second.t;
            continue;
        }
        if (!registry().count(n.op_type))
            throw std::runtime_error("TinyInfer: unsupported op '" + n.op_type + "' (node " + n.name + ")");
        runnable.push_back(std::move(n));
    }
    model_.nodes = std::move(runnable);

    for (auto& vi : model_.inputs) {
        input_names_.push_back(vi.name);
        input_shapes_.push_back(vi.shape);
    }
    for (auto& vi : model_.outputs) output_names_.push_back(vi.name);

    // Liveness: release each intermediate right after the last node that reads it.
    std::unordered_map<std::string, size_t> last_use;
    for (size_t i = 0; i < model_.nodes.size(); ++i)
        for (auto& name : model_.nodes[i].inputs)
            if (!name.empty()) last_use[name] = i;
    std::set<std::string> keep(output_names_.begin(), output_names_.end());
    release_after_.assign(model_.nodes.size(), {});
    for (auto& kv : last_use)
        if (!constants_.count(kv.first) && !keep.count(kv.first)) release_after_[kv.second].push_back(kv.first);
}

std::vector<std::string> Session::op_types() const {
    std::set<std::string> s;
    for (auto& n : model_.nodes) s.insert(n.op_type);
    return {s.begin(), s.end()};
}

std::vector<Tensor> Session::run(const Tensor& input) {
    if (input_names_.size() != 1) throw std::runtime_error("run(Tensor) requires a single-input model");
    return run(std::map<std::string, Tensor>{{input_names_[0], input}});
}

std::vector<Tensor> Session::run(const std::map<std::string, Tensor>& feeds) {
    std::unordered_map<std::string, Tensor> values;
    for (size_t i = 0; i < input_names_.size(); ++i) {
        auto it = feeds.find(input_names_[i]);
        if (it == feeds.end()) throw std::runtime_error("missing feed for input '" + input_names_[i] + "'");
        const Shape& expect = input_shapes_[i];
        const Shape& got = it->second.shape;
        bool ok = expect.empty() || expect.size() == got.size();
        for (size_t d = 0; ok && d < expect.size(); ++d) ok = expect[d] < 0 || expect[d] == got[d];
        if (!ok) throw std::runtime_error("input '" + input_names_[i] + "' expects shape " + shape_str(expect) +
                                          ", got " + shape_str(got));
        values[it->first] = it->second;
    }

    auto lookup = [&](const std::string& name) -> const Tensor* {
        if (name.empty()) return nullptr;  // optional input left blank
        if (auto it = values.find(name); it != values.end()) return &it->second;
        if (auto it = constants_.find(name); it != constants_.end()) return &it->second;
        throw std::runtime_error("value '" + name + "' is not available");
    };

    for (size_t i = 0; i < model_.nodes.size(); ++i) {
        const Node& n = model_.nodes[i];
        Inputs xs;
        for (auto& name : n.inputs) xs.push_back(lookup(name));
        const auto t0 = std::chrono::steady_clock::now();
        Tensor out = registry().at(n.op_type)(n, xs);
        if (profiling_) {
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            auto& p = profile_[n.op_type];
            p.op_type = n.op_type;
            p.calls += 1;
            p.total_ms += ms;
        }
        values[n.outputs.at(0)] = std::move(out);
        for (auto& dead : release_after_[i]) values.erase(dead);
    }

    std::vector<Tensor> outs;
    for (auto& name : output_names_) outs.push_back(*lookup(name));
    return outs;
}

std::vector<OpProfile> Session::profile() const {
    std::vector<OpProfile> v;
    for (auto& kv : profile_) v.push_back(kv.second);
    std::sort(v.begin(), v.end(), [](const OpProfile& a, const OpProfile& b) { return a.total_ms > b.total_ms; });
    return v;
}

}  // namespace tinyinfer
