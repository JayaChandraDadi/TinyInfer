#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "tinyinfer/onnx_reader.h"
#include "tinyinfer/ops.h"

using namespace tinyinfer;

namespace {
Tensor random_tensor(Shape s, uint32_t seed) {
    Tensor t(std::move(s));
    std::mt19937 rng(seed);
    std::normal_distribution<float> d;
    for (auto& v : t.data) v = d(rng);
    return t;
}

void expect_close(const Tensor& a, const Tensor& b, float tol) {
    ASSERT_EQ(a.shape, b.shape) << shape_str(a.shape) << " vs " << shape_str(b.shape);
    for (size_t i = 0; i < a.data.size(); ++i) ASSERT_NEAR(a.data[i], b.data[i], tol) << "at " << i;
}
}  // namespace

// ---------------------------------------------------------------------------
// Convolution: im2col + GEMM must agree with the direct reference everywhere.
// ---------------------------------------------------------------------------
struct ConvCase {
    int64_t n, c, h, w, m, k, stride, pad, dilation, group;
};

class ConvAgreement : public ::testing::TestWithParam<ConvCase> {};

TEST_P(ConvAgreement, Im2colMatchesDirect) {
    const ConvCase cc = GetParam();
    Tensor x = random_tensor({cc.n, cc.c, cc.h, cc.w}, 1);
    Tensor w = random_tensor({cc.m, cc.c / cc.group, cc.k, cc.k}, 2);
    Tensor b = random_tensor({cc.m}, 3);
    ops::Conv2dParams p;
    p.stride = {cc.stride, cc.stride};
    p.pads = {cc.pad, cc.pad, cc.pad, cc.pad};
    p.dilation = {cc.dilation, cc.dilation};
    p.group = cc.group;
    for (const Tensor* bias : std::vector<const Tensor*>{nullptr, &b}) {
        Tensor ref = ops::conv2d_direct(x, w, bias, p);
        Tensor fast = ops::conv2d_im2col(x, w, bias, p);
        expect_close(ref, fast, 1e-3f);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Params, ConvAgreement,
    ::testing::Values(ConvCase{1, 3, 8, 8, 4, 3, 1, 1, 1, 1},     // same padding
                      ConvCase{2, 3, 9, 7, 5, 3, 2, 0, 1, 1},     // stride 2, no pad, non-square
                      ConvCase{1, 8, 10, 10, 16, 1, 1, 0, 1, 1},  // pointwise fast path
                      ConvCase{1, 8, 10, 10, 16, 1, 2, 0, 1, 1},  // strided 1x1 (needs im2col)
                      ConvCase{1, 4, 12, 12, 6, 3, 1, 2, 2, 1},   // dilation
                      ConvCase{1, 8, 9, 9, 8, 5, 1, 2, 1, 8},     // depthwise
                      ConvCase{2, 8, 10, 9, 8, 3, 2, 1, 1, 8},    // depthwise, stride 2
                      ConvCase{1, 6, 11, 11, 6, 3, 1, 2, 2, 6},   // depthwise, dilated
                      ConvCase{1, 4, 3, 3, 4, 5, 1, 0, 1, 4},     // depthwise, kernel > input + pad
                      ConvCase{2, 8, 9, 9, 12, 3, 1, 1, 1, 4},    // grouped
                      ConvCase{1, 16, 15, 15, 32, 5, 2, 2, 1, 1}));

TEST(Conv, KnownValues) {
    // 1x1x3x3 input of ones, 3x3 kernel of ones, pad 1 -> corners 4, edges 6, centre 9.
    Tensor x({1, 1, 3, 3}, std::vector<float>(9, 1.0f));
    Tensor w({1, 1, 3, 3}, std::vector<float>(9, 1.0f));
    ops::Conv2dParams p;
    p.pads = {1, 1, 1, 1};
    Tensor y = ops::conv2d_im2col(x, w, nullptr, p);
    const std::vector<float> expect = {4, 6, 4, 6, 9, 6, 4, 6, 4};
    EXPECT_EQ(y.data, expect);
}

// ---------------------------------------------------------------------------
// Pooling
// ---------------------------------------------------------------------------
TEST(Pool, MaxPoolCeilModeDropsWindowStartingInPadding) {
    // PyTorch: MaxPool2d(3, stride=2, ceil_mode=True) on 32x32 -> 16x16 (not 17x17).
    EXPECT_EQ(ops::pool_out_dim(32, 3, 2, 0, 0, true), 16);
    EXPECT_EQ(ops::pool_out_dim(32, 3, 2, 0, 0, false), 15);
    EXPECT_EQ(ops::pool_out_dim(5, 2, 2, 0, 0, true), 3);
}

TEST(Pool, MaxPoolValues) {
    Tensor x({1, 1, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    ops::Pool2dParams p;
    p.kernel = {2, 2};
    p.stride = {2, 2};
    Tensor y = ops::maxpool2d(x, p);
    EXPECT_EQ(y.shape, (Shape{1, 1, 2, 2}));
    EXPECT_EQ(y.data, (std::vector<float>{6, 8, 14, 16}));
}

TEST(Pool, AvgPoolExcludesPaddingByDefault) {
    Tensor x({1, 1, 2, 2}, {1, 2, 3, 4});
    ops::Pool2dParams p;
    p.kernel = {2, 2};
    p.stride = {1, 1};
    p.pads = {1, 1, 0, 0};
    Tensor y = ops::avgpool2d(x, p);
    EXPECT_FLOAT_EQ(y.data[0], 1.0f);          // only the real pixel counts
    p.count_include_pad = true;
    EXPECT_FLOAT_EQ(ops::avgpool2d(x, p).data[0], 0.25f);
}

TEST(Pool, GlobalAverage) {
    Tensor x = random_tensor({2, 3, 5, 5}, 4);
    Tensor y = ops::global_avgpool(x);
    ASSERT_EQ(y.shape, (Shape{2, 3, 1, 1}));
    double s = 0;
    for (int i = 0; i < 25; ++i) s += x.data[static_cast<size_t>(i)];
    EXPECT_NEAR(y.data[0], s / 25.0, 1e-5);
}

// ---------------------------------------------------------------------------
// Element-wise, dense and shape ops
// ---------------------------------------------------------------------------
TEST(Activations, Values) {
    Tensor x({5}, {-4.0f, -1.0f, 0.0f, 1.0f, 4.0f});
    EXPECT_EQ(ops::relu(x).data, (std::vector<float>{0, 0, 0, 1, 4}));
    EXPECT_EQ(ops::clip(x, 0.0f, 3.0f).data, (std::vector<float>{0, 0, 0, 1, 3}));
    Tensor hs = ops::hardswish(x);
    EXPECT_FLOAT_EQ(hs.data[0], 0.0f);
    EXPECT_FLOAT_EQ(hs.data[3], 1.0f * (1.0f + 3.0f) / 6.0f);
    EXPECT_FLOAT_EQ(hs.data[4], 4.0f);
    EXPECT_NEAR(ops::sigmoid(x).data[2], 0.5f, 1e-7f);
}

TEST(Broadcast, ChannelScaleLikeSqueezeExcite) {
    Tensor x = random_tensor({2, 3, 4, 4}, 5);
    Tensor s = random_tensor({2, 3, 1, 1}, 6);
    Tensor y = ops::binary(x, s, ops::BinOp::Mul);
    ASSERT_EQ(y.shape, x.shape);
    for (int64_t n = 0; n < 2; ++n)
        for (int64_t c = 0; c < 3; ++c)
            for (int64_t i = 0; i < 16; ++i) {
                const size_t idx = static_cast<size_t>((n * 3 + c) * 16 + i);
                ASSERT_FLOAT_EQ(y.data[idx], x.data[idx] * s.data[static_cast<size_t>(n * 3 + c)]);
            }
}

TEST(Broadcast, RowVectorAndScalar) {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor row({3}, {10, 20, 30});
    EXPECT_EQ(ops::binary(a, row, ops::BinOp::Add).data, (std::vector<float>{11, 22, 33, 14, 25, 36}));
    Tensor scalar({}, {2.0f});
    EXPECT_EQ(ops::binary(a, scalar, ops::BinOp::Div).data, (std::vector<float>{0.5f, 1, 1.5f, 2, 2.5f, 3}));
}

TEST(Broadcast, IncompatibleShapesThrow) {
    EXPECT_THROW(ops::binary(Tensor({2, 3}), Tensor({4}), ops::BinOp::Add), std::runtime_error);
}

TEST(Dense, GemmTransBWithBiasMatchesManual) {
    Tensor x({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor w({2, 3}, {1, 0, -1, 0.5f, 0.5f, 0.5f});  // [out, in] like nn.Linear
    Tensor b({2}, {10, 20});
    Tensor y = ops::gemm(x, w, &b, false, true, 1.0f, 1.0f);
    EXPECT_EQ(y.shape, (Shape{2, 2}));
    EXPECT_EQ(y.data, (std::vector<float>{8, 23, 8, 27.5f}));
}

TEST(Dense, MatMulFoldsBatchDims) {
    Tensor a = random_tensor({2, 3, 4}, 7);
    Tensor b = random_tensor({4, 5}, 8);
    Tensor y = ops::matmul(a, b);
    ASSERT_EQ(y.shape, (Shape{2, 3, 5}));
    float s = 0;
    for (int k = 0; k < 4; ++k) s += a.data[static_cast<size_t>(4 + k)] * b.data[static_cast<size_t>(k * 5 + 2)];
    EXPECT_NEAR(y.data[5 + 2], s, 1e-5f);
}

TEST(Shape, FlattenReshapeSoftmax) {
    Tensor x = random_tensor({2, 3, 4, 5}, 9);
    EXPECT_EQ(ops::flatten(x, 1).shape, (Shape{2, 60}));
    EXPECT_EQ(ops::reshape(x, {0, -1}).shape, (Shape{2, 60}));
    EXPECT_EQ(ops::reshape(x, {2, 0, 20}).shape, (Shape{2, 3, 20}));  // 0 copies the input dim
    EXPECT_THROW(ops::reshape(x, {7, -1}), std::runtime_error);
    Tensor sm = ops::softmax(ops::flatten(x, 1), -1);
    for (int r = 0; r < 2; ++r) {
        float s = 0;
        for (int i = 0; i < 60; ++i) s += sm.data[static_cast<size_t>(r * 60 + i)];
        EXPECT_NEAR(s, 1.0f, 1e-5f);
    }
}

TEST(OnnxReader, RejectsGarbage) {
    EXPECT_THROW(parse_onnx(std::string("\xff\xff\xff", 3)), std::runtime_error);
    EXPECT_THROW(parse_onnx(std::string()), std::runtime_error);
}

TEST(Norm, BatchNormMatchesFormula) {
    // The PyTorch exporter folds BN into Conv in eval mode, so this op is only
    // reached by models exported with BN kept separate; test it directly.
    Tensor x = random_tensor({2, 3, 4, 4}, 10);
    Tensor scale({3}, {1.5f, 0.5f, 2.0f}), bias({3}, {0.1f, -0.2f, 0.0f});
    Tensor mean({3}, {0.3f, -0.1f, 1.0f}), var({3}, {0.8f, 1.2f, 2.5f});
    Tensor y = ops::batchnorm(x, scale, bias, mean, var, 1e-5f);
    for (int64_t n = 0; n < 2; ++n)
        for (int64_t c = 0; c < 3; ++c)
            for (int64_t i = 0; i < 16; ++i) {
                const size_t idx = static_cast<size_t>((n * 3 + c) * 16 + i);
                const size_t ci = static_cast<size_t>(c);
                const float ref = (x.data[idx] - mean.data[ci]) / std::sqrt(var.data[ci] + 1e-5f) * scale.data[ci] + bias.data[ci];
                ASSERT_NEAR(y.data[idx], ref, 1e-5f);
            }
}
