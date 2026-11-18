#include "tinyinfer/ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tinyinfer::ops {

namespace {
void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

int64_t conv_out_dim(int64_t in, int64_t k, int64_t s, int64_t pb, int64_t pe, int64_t d) {
    return (in + pb + pe - d * (k - 1) - 1) / s + 1;
}

void add_bias(Tensor& y, const Tensor* bias) {
    if (!bias) return;
    const int64_t N = y.dim(0), M = y.dim(1), HW = y.dim(2) * y.dim(3);
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n)
        for (int64_t m = 0; m < M; ++m) {
            float* p = y.ptr() + (n * M + m) * HW;
            const float b = bias->data[static_cast<size_t>(m)];
            for (int64_t i = 0; i < HW; ++i) p[i] += b;
        }
}

template <typename F>
Tensor map(const Tensor& x, F f) {
    Tensor y(x.shape);
    const int64_t n = x.size();
    const float* src = x.ptr();
    float* dst = y.ptr();
#pragma omp parallel for simd schedule(static) if (n > 16384)
    for (int64_t i = 0; i < n; ++i) dst[i] = f(src[i]);
    return y;
}
}  // namespace

int64_t pool_out_dim(int64_t in, int64_t k, int64_t s, int64_t pb, int64_t pe, bool ceil_mode, int64_t d) {
    const int64_t span = in + pb + pe - d * (k - 1) - 1;
    int64_t out = (ceil_mode ? (span + s - 1) / s : span / s) + 1;
    // PyTorch/ONNX rule: the last window must start inside the input or left padding.
    if (ceil_mode && (out - 1) * s >= in + pb) --out;
    return out;
}

// ---------------------------------------------------------------------------
// Convolution
// ---------------------------------------------------------------------------
void im2col(const float* img, int64_t C, int64_t H, int64_t W, int64_t kh, int64_t kw, const Conv2dParams& p,
            int64_t Hout, int64_t Wout, float* col) {
    const int64_t sh = p.stride[0], sw = p.stride[1];
    const int64_t dh = p.dilation[0], dw = p.dilation[1];
    const int64_t pt = p.pads[0], pl = p.pads[1];
    const int64_t HWo = Hout * Wout;
    const int64_t rows = C * kh * kw;
#pragma omp parallel for schedule(static) if (rows * HWo > 32768)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t c = r / (kh * kw);
        const int64_t ki = (r / kw) % kh;
        const int64_t kj = r % kw;
        float* dst = col + r * HWo;
        const float* src = img + c * H * W;
        for (int64_t oh = 0; oh < Hout; ++oh) {
            const int64_t ih = oh * sh - pt + ki * dh;
            float* drow = dst + oh * Wout;
            if (ih < 0 || ih >= H) {
                std::memset(drow, 0, sizeof(float) * static_cast<size_t>(Wout));
                continue;
            }
            const float* srow = src + ih * W;
            for (int64_t ow = 0; ow < Wout; ++ow) {
                const int64_t iw = ow * sw - pl + kj * dw;
                drow[ow] = (iw >= 0 && iw < W) ? srow[iw] : 0.0f;
            }
        }
    }
}

namespace {
// Depthwise conv (one input and one output channel per group): im2col would turn
// this into thousands of 1-row GEMMs, so it gets its own direct kernel.
Tensor conv2d_depthwise(const Tensor& x, const Tensor& w, const Tensor* bias, const Conv2dParams& p) {
    const int64_t N = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
    const int64_t kh = w.dim(2), kw = w.dim(3);
    const int64_t sh = p.stride[0], sw = p.stride[1], dh = p.dilation[0], dw = p.dilation[1];
    const int64_t pt = p.pads[0], pl = p.pads[1];
    const int64_t Ho = conv_out_dim(H, kh, sh, pt, p.pads[2], dh);
    const int64_t Wo = conv_out_dim(W, kw, sw, pl, p.pads[3], dw);
    // For kernel column j, the output columns whose input column lies inside the
    // image form one contiguous range; precompute it once instead of bounds-
    // checking every tap.
    std::vector<int64_t> ow_lo(static_cast<size_t>(kw)), ow_hi(static_cast<size_t>(kw));
    for (int64_t j = 0; j < kw; ++j) {
        const int64_t off = j * dw - pl;  // iw = ow * sw + off
        int64_t lo = off >= 0 ? 0 : (-off + sw - 1) / sw;
        int64_t hi = (W - 1 - off) >= 0 ? (W - 1 - off) / sw + 1 : 0;
        ow_lo[static_cast<size_t>(j)] = std::min(lo, Wo);
        ow_hi[static_cast<size_t>(j)] = std::min(hi, Wo);
    }
    Tensor y({N, C, Ho, Wo});
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c) {
            const float* src = x.ptr() + (n * C + c) * H * W;
            const float* k = w.ptr() + c * kh * kw;
            float* dst = y.ptr() + (n * C + c) * Ho * Wo;
            const float b = bias ? bias->data[static_cast<size_t>(c)] : 0.0f;
            for (int64_t oh = 0; oh < Ho; ++oh) {
                float* drow = dst + oh * Wo;
                for (int64_t ow = 0; ow < Wo; ++ow) drow[ow] = b;
                for (int64_t i = 0; i < kh; ++i) {
                    const int64_t ih = oh * sh - pt + i * dh;
                    if (ih < 0 || ih >= H) continue;
                    const float* srow = src + ih * W;
                    for (int64_t j = 0; j < kw; ++j) {
                        const float kv = k[i * kw + j];
                        const int64_t off = j * dw - pl;
                        const int64_t lo = ow_lo[static_cast<size_t>(j)], hi = ow_hi[static_cast<size_t>(j)];
                        if (sw == 1) {
                            const float* s1 = srow + off;
#pragma omp simd
                            for (int64_t ow = lo; ow < hi; ++ow) drow[ow] += kv * s1[ow];
                        } else {
                            for (int64_t ow = lo; ow < hi; ++ow) drow[ow] += kv * srow[ow * sw + off];
                        }
                    }
                }
            }
        }
    return y;
}
}  // namespace

Tensor conv2d_im2col(const Tensor& x, const Tensor& w, const Tensor* bias, const Conv2dParams& p) {
    require(x.rank() == 4 && w.rank() == 4, "Conv: expected 4-D input and weight");
    const int64_t N = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
    const int64_t M = w.dim(0), Cg = w.dim(1), kh = w.dim(2), kw = w.dim(3);
    const int64_t G = p.group;
    require(C == Cg * G && M % G == 0, "Conv: channel/group mismatch");
    const int64_t Mg = M / G;
    if (Cg == 1 && Mg == 1) return conv2d_depthwise(x, w, bias, p);

    const int64_t Ho = conv_out_dim(H, kh, p.stride[0], p.pads[0], p.pads[2], p.dilation[0]);
    const int64_t Wo = conv_out_dim(W, kw, p.stride[1], p.pads[1], p.pads[3], p.dilation[1]);
    const int64_t HWo = Ho * Wo, K = Cg * kh * kw;
    Tensor y({N, M, Ho, Wo});

    // 1x1/stride-1/no-pad convs are already a GEMM over the input: skip the unfold.
    const bool pointwise = kh == 1 && kw == 1 && p.stride[0] == 1 && p.stride[1] == 1 &&
                           p.pads[0] == 0 && p.pads[1] == 0 && p.pads[2] == 0 && p.pads[3] == 0;
    std::vector<float> col(pointwise ? 0 : static_cast<size_t>(K * HWo));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < G; ++g) {
            const float* img = x.ptr() + (n * C + g * Cg) * H * W;
            const float* B = img;
            if (!pointwise) {
                im2col(img, Cg, H, W, kh, kw, p, Ho, Wo, col.data());
                B = col.data();
            }
            tinyinfer::gemm(Mg, HWo, K, w.ptr() + g * Mg * K, B, y.ptr() + (n * M + g * Mg) * HWo);
        }
    add_bias(y, bias);
    return y;
}

// Reference implementation: seven nested loops, used by tests and as the
// benchmark baseline.
Tensor conv2d_direct(const Tensor& x, const Tensor& w, const Tensor* bias, const Conv2dParams& p) {
    const int64_t N = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
    const int64_t M = w.dim(0), Cg = w.dim(1), kh = w.dim(2), kw = w.dim(3);
    const int64_t G = p.group;
    require(C == Cg * G && M % G == 0, "Conv: channel/group mismatch");
    const int64_t Mg = M / G;
    const int64_t Ho = conv_out_dim(H, kh, p.stride[0], p.pads[0], p.pads[2], p.dilation[0]);
    const int64_t Wo = conv_out_dim(W, kw, p.stride[1], p.pads[1], p.pads[3], p.dilation[1]);
    Tensor y({N, M, Ho, Wo});
    for (int64_t n = 0; n < N; ++n)
        for (int64_t m = 0; m < M; ++m) {
            const int64_t g = m / Mg;
            for (int64_t oh = 0; oh < Ho; ++oh)
                for (int64_t ow = 0; ow < Wo; ++ow) {
                    float acc = bias ? bias->data[static_cast<size_t>(m)] : 0.0f;
                    for (int64_t c = 0; c < Cg; ++c)
                        for (int64_t i = 0; i < kh; ++i) {
                            const int64_t ih = oh * p.stride[0] - p.pads[0] + i * p.dilation[0];
                            if (ih < 0 || ih >= H) continue;
                            for (int64_t j = 0; j < kw; ++j) {
                                const int64_t iw = ow * p.stride[1] - p.pads[1] + j * p.dilation[1];
                                if (iw < 0 || iw >= W) continue;
                                acc += x.data[static_cast<size_t>(((n * C + g * Cg + c) * H + ih) * W + iw)] *
                                       w.data[static_cast<size_t>(((m * Cg + c) * kh + i) * kw + j)];
                            }
                        }
                    y.data[static_cast<size_t>(((n * M + m) * Ho + oh) * Wo + ow)] = acc;
                }
        }
    return y;
}

Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor* bias, const Conv2dParams& p, ConvAlgo algo) {
    return algo == ConvAlgo::Direct ? conv2d_direct(x, w, bias, p) : conv2d_im2col(x, w, bias, p);
}

// ---------------------------------------------------------------------------
// Pooling
// ---------------------------------------------------------------------------
Tensor maxpool2d(const Tensor& x, const Pool2dParams& p) {
    require(x.rank() == 4, "MaxPool: expected 4-D input");
    const int64_t N = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
    const int64_t Ho = pool_out_dim(H, p.kernel[0], p.stride[0], p.pads[0], p.pads[2], p.ceil_mode);
    const int64_t Wo = pool_out_dim(W, p.kernel[1], p.stride[1], p.pads[1], p.pads[3], p.ceil_mode);
    Tensor y({N, C, Ho, Wo});
#pragma omp parallel for schedule(static)
    for (int64_t nc = 0; nc < N * C; ++nc) {
        const float* src = x.ptr() + nc * H * W;
        float* dst = y.ptr() + nc * Ho * Wo;
        for (int64_t oh = 0; oh < Ho; ++oh) {
            const int64_t h0 = std::max<int64_t>(oh * p.stride[0] - p.pads[0], 0);
            const int64_t h1 = std::min<int64_t>(oh * p.stride[0] - p.pads[0] + p.kernel[0], H);
            for (int64_t ow = 0; ow < Wo; ++ow) {
                const int64_t w0 = std::max<int64_t>(ow * p.stride[1] - p.pads[1], 0);
                const int64_t w1 = std::min<int64_t>(ow * p.stride[1] - p.pads[1] + p.kernel[1], W);
                float m = -std::numeric_limits<float>::infinity();
                for (int64_t ih = h0; ih < h1; ++ih)
                    for (int64_t iw = w0; iw < w1; ++iw) m = std::max(m, src[ih * W + iw]);
                dst[oh * Wo + ow] = m;
            }
        }
    }
    return y;
}

Tensor avgpool2d(const Tensor& x, const Pool2dParams& p) {
    require(x.rank() == 4, "AveragePool: expected 4-D input");
    const int64_t N = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
    const int64_t Ho = pool_out_dim(H, p.kernel[0], p.stride[0], p.pads[0], p.pads[2], p.ceil_mode);
    const int64_t Wo = pool_out_dim(W, p.kernel[1], p.stride[1], p.pads[1], p.pads[3], p.ceil_mode);
    Tensor y({N, C, Ho, Wo});
#pragma omp parallel for schedule(static)
    for (int64_t nc = 0; nc < N * C; ++nc) {
        const float* src = x.ptr() + nc * H * W;
        float* dst = y.ptr() + nc * Ho * Wo;
        for (int64_t oh = 0; oh < Ho; ++oh) {
            const int64_t hs = oh * p.stride[0] - p.pads[0];
            const int64_t he_pad = std::min<int64_t>(hs + p.kernel[0], H + p.pads[2]);
            const int64_t h0 = std::max<int64_t>(hs, 0), h1 = std::min<int64_t>(hs + p.kernel[0], H);
            for (int64_t ow = 0; ow < Wo; ++ow) {
                const int64_t ws = ow * p.stride[1] - p.pads[1];
                const int64_t we_pad = std::min<int64_t>(ws + p.kernel[1], W + p.pads[3]);
                const int64_t w0 = std::max<int64_t>(ws, 0), w1 = std::min<int64_t>(ws + p.kernel[1], W);
                float s = 0.0f;
                for (int64_t ih = h0; ih < h1; ++ih)
                    for (int64_t iw = w0; iw < w1; ++iw) s += src[ih * W + iw];
                const int64_t count = p.count_include_pad ? (he_pad - hs) * (we_pad - ws) : (h1 - h0) * (w1 - w0);
                dst[oh * Wo + ow] = count > 0 ? s / static_cast<float>(count) : 0.0f;
            }
        }
    }
    return y;
}

Tensor global_avgpool(const Tensor& x) {
    require(x.rank() == 4, "GlobalAveragePool: expected 4-D input");
    const int64_t N = x.dim(0), C = x.dim(1), HW = x.dim(2) * x.dim(3);
    Tensor y({N, C, 1, 1});
#pragma omp parallel for schedule(static) if (N * C * HW > 32768)
    for (int64_t nc = 0; nc < N * C; ++nc) {
        const float* src = x.ptr() + nc * HW;
        double s = 0.0;
        for (int64_t i = 0; i < HW; ++i) s += src[i];
        y.data[static_cast<size_t>(nc)] = static_cast<float>(s / static_cast<double>(HW));
    }
    return y;
}

// ---------------------------------------------------------------------------
// Activations / normalisation
// ---------------------------------------------------------------------------
Tensor relu(const Tensor& x) { return map(x, [](float v) { return v > 0.0f ? v : 0.0f; }); }
Tensor clip(const Tensor& x, float lo, float hi) {
    return map(x, [lo, hi](float v) { return std::min(std::max(v, lo), hi); });
}
Tensor sigmoid(const Tensor& x) { return map(x, [](float v) { return 1.0f / (1.0f + std::exp(-v)); }); }
Tensor hardsigmoid(const Tensor& x, float alpha, float beta) {
    return map(x, [alpha, beta](float v) { return std::min(std::max(alpha * v + beta, 0.0f), 1.0f); });
}
Tensor hardswish(const Tensor& x) {
    return map(x, [](float v) { return v * std::min(std::max(v / 6.0f + 0.5f, 0.0f), 1.0f); });
}

Tensor batchnorm(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& mean, const Tensor& var,
                 float eps) {
    const int64_t N = x.dim(0), C = x.dim(1);
    const int64_t inner = x.size() / (N * C);
    Tensor y(x.shape);
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c) {
            const size_t ci = static_cast<size_t>(c);
            const float a = scale.data[ci] / std::sqrt(var.data[ci] + eps);
            const float b = bias.data[ci] - mean.data[ci] * a;
            const float* s = x.ptr() + (n * C + c) * inner;
            float* d = y.ptr() + (n * C + c) * inner;
            for (int64_t i = 0; i < inner; ++i) d[i] = s[i] * a + b;
        }
    return y;
}

// ---------------------------------------------------------------------------
// Dense layers
// ---------------------------------------------------------------------------
namespace {
std::vector<float> transpose2d(const float* src, int64_t rows, int64_t cols) {
    std::vector<float> t(static_cast<size_t>(rows * cols));
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c) t[static_cast<size_t>(c * rows + r)] = src[r * cols + c];
    return t;
}
}  // namespace

Tensor gemm(const Tensor& A, const Tensor& B, const Tensor* C, bool transA, bool transB, float alpha, float beta) {
    require(A.rank() == 2 && B.rank() == 2, "Gemm: expected 2-D operands");
    const int64_t M = transA ? A.dim(1) : A.dim(0);
    const int64_t K = transA ? A.dim(0) : A.dim(1);
    const int64_t Kb = transB ? B.dim(1) : B.dim(0);
    const int64_t N = transB ? B.dim(0) : B.dim(1);
    require(K == Kb, "Gemm: inner dimensions differ " + shape_str(A.shape) + " x " + shape_str(B.shape));

    std::vector<float> at, bt;
    const float* a = A.ptr();
    if (transA) { at = transpose2d(A.ptr(), A.dim(0), A.dim(1)); a = at.data(); }

    Tensor y({M, N});
    if (transB && M <= 16) {
        // Fully-connected layer at small batch (nn.Linear stores W as [out, in]):
        // each output is a contiguous dot product of an input row with a weight
        // row, so skip transposing W on every call.
        const float* b = B.ptr();
#pragma omp parallel for collapse(2) schedule(static) if (M * N * K > 32768)
        for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < N; ++j) {
                const float* ar = a + i * K;
                const float* br = b + j * K;
                float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
                for (int64_t k = 0; k < K; ++k) acc += ar[k] * br[k];
                y.data[static_cast<size_t>(i * N + j)] = acc;
            }
    } else {
        const float* b = B.ptr();
        if (transB) { bt = transpose2d(B.ptr(), B.dim(0), B.dim(1)); b = bt.data(); }
        tinyinfer::gemm(M, N, K, a, b, y.ptr());
    }
    if (alpha != 1.0f)
        for (auto& v : y.data) v *= alpha;
    if (C && beta != 0.0f) {
        const int64_t cs = C->size();
        for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < N; ++j) {
                float c;
                if (cs == 1) c = C->data[0];
                else if (cs == N) c = C->data[static_cast<size_t>(j)];
                else if (cs == M * N) c = C->data[static_cast<size_t>(i * N + j)];
                else if (cs == M) c = C->data[static_cast<size_t>(i)];
                else throw std::runtime_error("Gemm: cannot broadcast C " + shape_str(C->shape));
                y.data[static_cast<size_t>(i * N + j)] += beta * c;
            }
    }
    return y;
}

Tensor matmul(const Tensor& A, const Tensor& B) {
    require(B.rank() == 2 && A.rank() >= 2, "MatMul: only [..., M, K] x [K, N] is supported");
    const int64_t K = A.shape.back();
    require(K == B.dim(0), "MatMul: inner dimensions differ");
    const int64_t N = B.dim(1);
    const int64_t rows = A.size() / K;  // fold batch dims into M
    Shape out = A.shape;
    out.back() = N;
    Tensor y(out);
    tinyinfer::gemm(rows, N, K, A.ptr(), B.ptr(), y.ptr());
    return y;
}

// ---------------------------------------------------------------------------
// Shape ops
// ---------------------------------------------------------------------------
Tensor flatten(const Tensor& x, int64_t axis) {
    if (axis < 0) axis += static_cast<int64_t>(x.rank());
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= x.shape[static_cast<size_t>(i)];
    return Tensor({outer, x.size() / outer}, x.data);
}

Tensor reshape(const Tensor& x, const std::vector<int64_t>& shape, bool allowzero) {
    Shape out(shape.begin(), shape.end());
    int64_t known = 1, infer = -1;
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] == 0 && !allowzero) out[i] = x.shape.at(i);
        if (out[i] == -1) infer = static_cast<int64_t>(i);
        else known *= out[i];
    }
    if (infer >= 0) out[static_cast<size_t>(infer)] = x.size() / known;
    require(numel(out) == x.size(), "Reshape: cannot reshape " + shape_str(x.shape) + " to " + shape_str(out));
    if (x.is_int) return Tensor::ints(out, x.i64);
    return Tensor(out, x.data);
}

Tensor softmax(const Tensor& x, int64_t axis) {
    const int64_t r = static_cast<int64_t>(x.rank());
    if (axis < 0) axis += r;
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= x.shape[static_cast<size_t>(i)];
    for (int64_t i = axis + 1; i < r; ++i) inner *= x.shape[static_cast<size_t>(i)];
    const int64_t n = x.shape[static_cast<size_t>(axis)];
    Tensor y(x.shape);
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t in = 0; in < inner; ++in) {
            const float* s = x.ptr() + o * n * inner + in;
            float* d = y.ptr() + o * n * inner + in;
            float m = -std::numeric_limits<float>::infinity();
            for (int64_t i = 0; i < n; ++i) m = std::max(m, s[i * inner]);
            float sum = 0.0f;
            for (int64_t i = 0; i < n; ++i) sum += (d[i * inner] = std::exp(s[i * inner] - m));
            for (int64_t i = 0; i < n; ++i) d[i * inner] /= sum;
        }
    return y;
}

// ---------------------------------------------------------------------------
// Broadcasting element-wise ops
// ---------------------------------------------------------------------------
Tensor binary(const Tensor& a, const Tensor& b, BinOp op) {
    auto apply = [op](float x, float y) {
        switch (op) {
            case BinOp::Add: return x + y;
            case BinOp::Sub: return x - y;
            case BinOp::Mul: return x * y;
            case BinOp::Div: return x / y;
        }
        return 0.0f;
    };

    if (a.shape == b.shape) {
        Tensor y(a.shape);
        const int64_t n = a.size();
#pragma omp parallel for schedule(static) if (n > 16384)
        for (int64_t i = 0; i < n; ++i) y.data[static_cast<size_t>(i)] = apply(a.data[static_cast<size_t>(i)], b.data[static_cast<size_t>(i)]);
        return y;
    }

    const size_t r = std::max(a.rank(), b.rank());
    Shape as(r, 1), bs(r, 1), out(r, 1);
    std::copy(a.shape.begin(), a.shape.end(), as.begin() + static_cast<long>(r - a.rank()));
    std::copy(b.shape.begin(), b.shape.end(), bs.begin() + static_cast<long>(r - b.rank()));
    for (size_t i = 0; i < r; ++i) {
        require(as[i] == bs[i] || as[i] == 1 || bs[i] == 1,
                "broadcast: incompatible shapes " + shape_str(a.shape) + " and " + shape_str(b.shape));
        out[i] = std::max(as[i], bs[i]);
    }
    // Strides that are 0 along broadcast dims.
    std::vector<int64_t> sa(r), sb(r);
    int64_t ka = 1, kb = 1;
    for (size_t i = r; i-- > 0;) {
        sa[i] = as[i] == 1 ? 0 : ka;
        sb[i] = bs[i] == 1 ? 0 : kb;
        ka *= as[i];
        kb *= bs[i];
    }
    Tensor y(out);
    const int64_t inner = out.back();
    const int64_t outer = y.size() / inner;
#pragma omp parallel for schedule(static) if (y.size() > 16384)
    for (int64_t o = 0; o < outer; ++o) {
        int64_t rem = o, ia = 0, ib = 0;
        for (size_t d = r - 1; d-- > 0;) {
            const int64_t idx = rem % out[d];
            rem /= out[d];
            ia += idx * sa[d];
            ib += idx * sb[d];
        }
        float* dst = y.ptr() + o * inner;
        for (int64_t j = 0; j < inner; ++j)
            dst[j] = apply(a.data[static_cast<size_t>(ia + j * sa[r - 1])], b.data[static_cast<size_t>(ib + j * sb[r - 1])]);
    }
    return y;
}

}  // namespace tinyinfer::ops
