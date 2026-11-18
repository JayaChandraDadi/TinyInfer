#pragma once

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tinyinfer {

using Shape = std::vector<int64_t>;

inline int64_t numel(const Shape& s) {
    return std::accumulate(s.begin(), s.end(), int64_t{1}, std::multiplies<int64_t>());
}

std::string shape_str(const Shape& s);

// Dense row-major float32 tensor. Integer tensors coming from ONNX (e.g. Reshape
// shapes) are kept in `i64` so that shape arithmetic never goes through floats.
struct Tensor {
    Shape shape;
    std::vector<float> data;
    std::vector<int64_t> i64;
    bool is_int = false;

    Tensor() = default;
    explicit Tensor(Shape s) : shape(std::move(s)), data(static_cast<size_t>(numel(shape)), 0.0f) {}
    Tensor(Shape s, std::vector<float> d) : shape(std::move(s)), data(std::move(d)) {
        if (static_cast<int64_t>(data.size()) != numel(shape))
            throw std::runtime_error("Tensor: data size does not match shape " + shape_str(shape));
    }

    static Tensor ints(Shape s, std::vector<int64_t> v) {
        Tensor t;
        t.shape = std::move(s);
        t.i64 = std::move(v);
        t.is_int = true;
        return t;
    }

    int64_t size() const { return is_int ? static_cast<int64_t>(i64.size()) : static_cast<int64_t>(data.size()); }
    int64_t dim(size_t i) const { return shape.at(i); }
    size_t rank() const { return shape.size(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

}  // namespace tinyinfer
