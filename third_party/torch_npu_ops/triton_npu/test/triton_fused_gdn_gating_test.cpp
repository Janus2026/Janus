/* Copyright 2026 The Janus Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <array>

#include "kernel_registry.h"
#include "test/test_utils.h"
#include "torch_api/triton_ops_api.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace janus::kernel::npu {

constexpr float kTolerance = 1e-3f;
constexpr int32_t kDeviceId = 0;
constexpr std::array<int64_t, 10> kNumHeadsTestCases =
    {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};

std::pair<torch::Tensor, torch::Tensor> torch_fused_gdn_gating(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& dt_bias,
    float beta = 1.0f,
    float threshold = 20.0f) {
  auto A_log_float = A_log.to(torch::kFloat32);
  auto a_float = a.to(torch::kFloat32);
  auto dt_bias_broadcast = dt_bias.unsqueeze(0).expand_as(a_float);
  auto a_plus_bias = a_float + dt_bias_broadcast;

  auto softplus_nn = torch::nn::Softplus(
      torch::nn::SoftplusOptions().beta(beta).threshold(threshold));
  auto softplus_res = softplus_nn(a_plus_bias);

  auto A_log_exp_broadcast =
      A_log_float.exp().unsqueeze(0).expand_as(softplus_res);

  auto g = -A_log_exp_broadcast * softplus_res;
  g = g.unsqueeze(0);

  auto beta_output = b.sigmoid().unsqueeze(0);

  return std::make_pair(g, beta_output);
}

class TritonFusedGdnGatingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kFloat16).device("npu:0");
      npu_available_ = true;
    } catch (...) {
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
      npu_available_ = false;
      return;
    }

    kernel_name_ = "fused_gdn_gating_decode_kernel";
    binary_filename_ = "fused_gdn_gating_decode_kernel.npubin";

    torch::manual_seed(42);
    torch_npu::init_npu(device_str_);

    binary_path_ = GetKernelBinaryPath(binary_filename_);
    auto& reg = KernelRegistry::get_instance();
    ASSERT_TRUE(reg.register_kernel(kernel_name_, binary_path_) ||
                reg.is_kernel_registered(kernel_name_))
        << "Failed to register kernel: " << kernel_name_ << " from "
        << binary_path_;
    ASSERT_NE(reg.get_kernel_stub(kernel_name_), nullptr)
        << "Failed to get kernel stub: " << kernel_name_;
  }

  void TearDown() override {
    if (npu_available_) {
      try {
        torch_npu::finalize_npu();
      } catch (...) {
      }
    }
  }

  torch::TensorOptions tensor_options_;
  bool npu_available_ = false;
  std::string device_str_ = "npu:" + std::to_string(kDeviceId);
  std::string binary_filename_;
  std::string kernel_name_;
  std::string binary_path_;
};

TEST_F(TritonFusedGdnGatingTest, DecodeKernelTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  auto device = at::Device(device_str_);
  constexpr int64_t num_tokens = 3;
  constexpr float beta = 1.0f;
  constexpr float threshold = 20.0f;

  for (int64_t num_heads : kNumHeadsTestCases) {
    auto A_log = torch::randn(
        {num_heads},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto a = torch::randn(
        {num_tokens, num_heads},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    auto b = torch::randn(
        {num_tokens, num_heads},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    auto dt_bias = torch::randn(
        {num_heads},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    auto A_log_cpu = A_log.cpu();
    auto a_cpu = a.cpu();
    auto b_cpu = b.cpu();
    auto dt_bias_cpu = dt_bias.cpu();
    auto [torch_g, torch_beta] = torch_fused_gdn_gating(
        A_log_cpu, a_cpu, b_cpu, dt_bias_cpu, beta, threshold);

    auto npu_stream = c10_npu::getCurrentNPUStream(0);
    auto [triton_g, triton_beta] =
        npu_fused_gdn_gating(A_log, a, b, dt_bias, beta, threshold);
    aclrtSynchronizeStream(npu_stream.stream());

    auto triton_g_cpu = triton_g.cpu();
    auto triton_beta_cpu = triton_beta.cpu();

    auto g_diff = torch::abs(torch_g - triton_g_cpu);
    float g_max_diff = torch::max(g_diff).item<float>();
    EXPECT_LT(g_max_diff, kTolerance)
        << "g: max diff (" << g_max_diff << ") > tolerance (" << kTolerance
        << ") for num_heads=" << num_heads;

    auto beta_diff = torch::abs(torch_beta - triton_beta_cpu);
    float beta_max_diff = torch::max(beta_diff).item<float>();
    EXPECT_LT(beta_max_diff, kTolerance)
        << "beta: max diff (" << beta_max_diff << ") > tolerance ("
        << kTolerance << ") for num_heads=" << num_heads;
  }
}

}  // namespace janus::kernel::npu

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  bool npu_available = false;
  std::string device_str =
      "npu:" + std::to_string(janus::kernel::npu::kDeviceId);
  try {
    auto test_tensor =
        torch::zeros({1}, torch::TensorOptions().device(device_str));
    (void)test_tensor;
    npu_available = true;
  } catch (...) {
    npu_available = false;
  }

  if (!npu_available) {
    LOG(WARNING) << "NPU device not available, skipping all tests.";
    return 0;
  }

  int result = RUN_ALL_TESTS();
  return result;
}
