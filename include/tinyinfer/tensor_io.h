#pragma once

// Minimal binary tensor format shared by the C++ tools/tests and the Python
// fixture generator:  int32 rank | int64 dims[rank] | float32 data[prod(dims)]

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

#include "tinyinfer/tensor.h"

namespace tinyinfer {

inline Tensor read_tensor(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    int32_t rank = 0;
    f.read(reinterpret_cast<char*>(&rank), sizeof(rank));
    Shape shape(static_cast<size_t>(rank));
    f.read(reinterpret_cast<char*>(shape.data()), static_cast<std::streamsize>(sizeof(int64_t) * shape.size()));
    Tensor t(shape);
    f.read(reinterpret_cast<char*>(t.ptr()), static_cast<std::streamsize>(sizeof(float) * t.data.size()));
    if (!f) throw std::runtime_error("truncated tensor file " + path);
    return t;
}

inline void write_tensor(const std::string& path, const Tensor& t) {
    std::ofstream f(path, std::ios::binary);
    const int32_t rank = static_cast<int32_t>(t.rank());
    f.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
    f.write(reinterpret_cast<const char*>(t.shape.data()), static_cast<std::streamsize>(sizeof(int64_t) * t.rank()));
    f.write(reinterpret_cast<const char*>(t.ptr()), static_cast<std::streamsize>(sizeof(float) * t.data.size()));
}

}  // namespace tinyinfer
