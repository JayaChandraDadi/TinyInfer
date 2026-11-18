// End-to-end check: each fixture holds an ONNX model exported from PyTorch, an
// input, and the output PyTorch produced for it (see scripts/gen_fixtures.py).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>

#include "tinyinfer/gemm.h"
#include "tinyinfer/session.h"
#include "tinyinfer/tensor_io.h"

using namespace tinyinfer;

namespace {
const std::string kFixtures = TINYINFER_FIXTURES;

float max_abs_diff(const Tensor& a, const Tensor& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.data.size(); ++i) m = std::max(m, std::fabs(a.data[i] - b.data[i]));
    return m;
}

struct ConfigGuard {
    EngineConfig saved = config();
    ~ConfigGuard() { config() = saved; }
};
}  // namespace

class PyTorchParity : public ::testing::TestWithParam<std::string> {};

TEST_P(PyTorchParity, MatchesPyTorchOutput) {
    const std::string dir = kFixtures + "/" + GetParam();
    Session s(dir + "/model.onnx");
    Tensor x = read_tensor(dir + "/input.bin");
    Tensor expected = read_tensor(dir + "/expected.bin");
    auto out = s.run(x);
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].shape, expected.shape);
    EXPECT_LT(max_abs_diff(out[0], expected), 1e-4f) << GetParam();
}

TEST_P(PyTorchParity, AllKernelConfigurationsAgree) {
    ConfigGuard guard;
    const std::string dir = kFixtures + "/" + GetParam();
    Session s(dir + "/model.onnx");
    Tensor x = read_tensor(dir + "/input.bin");
    Tensor expected = read_tensor(dir + "/expected.bin");
    for (ConvAlgo c : {ConvAlgo::Direct, ConvAlgo::Im2colGemm})
        for (GemmKernel g : {GemmKernel::Naive, GemmKernel::Blocked, GemmKernel::Simd}) {
            config().conv_algo = c;
            config().gemm_kernel = g;
            EXPECT_LT(max_abs_diff(s.run(x)[0], expected), 1e-4f)
                << GetParam() << " conv=" << static_cast<int>(c) << " gemm=" << static_cast<int>(g);
        }
}

INSTANTIATE_TEST_SUITE_P(Models, PyTorchParity,
                         ::testing::Values("lenet5", "small_cnn", "mini_resnet", "depthwise_se"));

TEST(Session, RejectsWrongInputShape) {
    Session s(kFixtures + "/lenet5/model.onnx");
    EXPECT_THROW(s.run(Tensor({1, 3, 28, 28})), std::runtime_error);
}

TEST(Session, ReportsOpsAndProfiles) {
    Session s(kFixtures + "/mini_resnet/model.onnx");
    auto ops = s.op_types();
    EXPECT_NE(std::find(ops.begin(), ops.end(), "Conv"), ops.end());
    EXPECT_NE(std::find(ops.begin(), ops.end(), "Add"), ops.end());
    s.enable_profiling(true);
    s.run(read_tensor(kFixtures + "/mini_resnet/input.bin"));
    auto prof = s.profile();
    ASSERT_FALSE(prof.empty());
    EXPECT_GT(prof.front().total_ms, 0.0);
}

TEST(Session, RunsRepeatedlyWithSameResult) {
    Session s(kFixtures + "/small_cnn/model.onnx");
    Tensor x = read_tensor(kFixtures + "/small_cnn/input.bin");
    auto a = s.run(x)[0];
    auto b = s.run(x)[0];
    EXPECT_EQ(a.data, b.data);
}
