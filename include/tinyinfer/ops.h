#pragma once

#include <array>
#include <vector>

#include "tinyinfer/gemm.h"
#include "tinyinfer/tensor.h"

namespace tinyinfer::ops {

struct Conv2dParams {
    std::array<int64_t, 2> stride{1, 1};
    std::array<int64_t, 4> pads{0, 0, 0, 0};  // top, left, bottom, right (ONNX order)
    std::array<int64_t, 2> dilation{1, 1};
    int64_t group = 1;
};

// x: [N, C, H, W], w: [M, C/group, kh, kw], bias: [M] or empty.
Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor* bias, const Conv2dParams& p,
              ConvAlgo algo = config().conv_algo);
Tensor conv2d_direct(const Tensor& x, const Tensor& w, const Tensor* bias, const Conv2dParams& p);
Tensor conv2d_im2col(const Tensor& x, const Tensor& w, const Tensor* bias, const Conv2dParams& p);

// Unfolds one image/group [C, H, W] into columns [C*kh*kw, Hout*Wout].
void im2col(const float* img, int64_t C, int64_t H, int64_t W, int64_t kh, int64_t kw, const Conv2dParams& p,
            int64_t Hout, int64_t Wout, float* col);

struct Pool2dParams {
    std::array<int64_t, 2> kernel{1, 1};
    std::array<int64_t, 2> stride{1, 1};
    std::array<int64_t, 4> pads{0, 0, 0, 0};
    bool ceil_mode = false;
    bool count_include_pad = false;
};
Tensor maxpool2d(const Tensor& x, const Pool2dParams& p);
Tensor avgpool2d(const Tensor& x, const Pool2dParams& p);
Tensor global_avgpool(const Tensor& x);

Tensor relu(const Tensor& x);
Tensor clip(const Tensor& x, float lo, float hi);
Tensor sigmoid(const Tensor& x);
Tensor hardsigmoid(const Tensor& x, float alpha = 1.0f / 6.0f, float beta = 0.5f);
Tensor hardswish(const Tensor& x);

Tensor batchnorm(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& mean, const Tensor& var,
                 float eps);

// Y = alpha * op(A) * op(B) + beta * C   (ONNX Gemm, 2-D only)
Tensor gemm(const Tensor& A, const Tensor& B, const Tensor* C, bool transA, bool transB, float alpha, float beta);
Tensor matmul(const Tensor& A, const Tensor& B);

Tensor flatten(const Tensor& x, int64_t axis);
Tensor reshape(const Tensor& x, const std::vector<int64_t>& shape, bool allowzero = false);
Tensor softmax(const Tensor& x, int64_t axis);

enum class BinOp { Add, Sub, Mul, Div };
Tensor binary(const Tensor& a, const Tensor& b, BinOp op);  // numpy-style broadcasting

int64_t pool_out_dim(int64_t in, int64_t k, int64_t s, int64_t pad_begin, int64_t pad_end, bool ceil_mode,
                     int64_t dilation = 1);

}  // namespace tinyinfer::ops
