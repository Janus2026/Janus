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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <vector>

#include "kernel_registry.h"
#include "test/test_utils.h"
#include "torch_api/triton_ops_api.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace janus::kernel::npu {

constexpr int32_t kDeviceId = 0;
constexpr float kTolerance = 1e-2f;  // bfloat16 tolerance

torch::Tensor rope_ref_cpu(const torch::Tensor& x,
                           const torch::Tensor& sin,
                           const torch::Tensor& cos,
                           int64_t rope_dim) {
  const int64_t total_dim = x.size(-1);
  const auto orig_dtype = x.dtype();

  torch::Tensor pre;
  if (rope_dim < total_dim) {
    pre = x.slice(-1, 0, total_dim - rope_dim);  // [..., total_dim - rope_dim]
  }

  auto x_rope =
      x.slice(-1, total_dim - rope_dim, total_dim);  // [..., rope_dim]

  torch::Tensor sin_exp = sin;
  torch::Tensor cos_exp = cos;
  if (sin.dim() == 2) {
    // (batch, rope_dim) -> (batch, 1, rope_dim), broadcast in head dimension
    sin_exp = sin.unsqueeze(1);
    cos_exp = cos.unsqueeze(1);
  }

  sin_exp = sin_exp.to(torch::kFloat32);
  cos_exp = cos_exp.to(torch::kFloat32);
  auto x_rope_f = x_rope.to(torch::kFloat32);

  auto idx_even = torch::arange(
      0,
      rope_dim,
      2,
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
  auto idx_odd = torch::arange(
      1,
      rope_dim,
      2,
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));

  auto x_even = x_rope_f.index_select(-1, idx_even);
  auto x_odd = x_rope_f.index_select(-1, idx_odd);

  auto x_rotate_f = torch::empty_like(x_rope_f);
  x_rotate_f.index_copy_(-1, idx_even, -x_odd);
  x_rotate_f.index_copy_(-1, idx_odd, x_even);

  auto out_rope = x_rope_f * cos_exp + x_rotate_f * sin_exp;
  out_rope = out_rope.to(orig_dtype);

  if (pre.defined()) {
    return torch::cat({pre, out_rope}, -1);
  }
  return out_rope;
}

class TritonRopeInplaceTest
    : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {
 protected:
  static bool npu_initialized_;

  static void SetUpTestSuite() {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      torch_npu::init_npu("npu:" + std::to_string(kDeviceId));
      auto& reg = KernelRegistry::get_instance();
      std::string binary_path =
          GetKernelBinaryPath("rope_inplace_kernel.npubin");
      npu_initialized_ =
          reg.register_kernel("rope_inplace_kernel", binary_path) &&
          reg.get_kernel_stub("rope_inplace_kernel") != nullptr;
    } catch (...) {
      npu_initialized_ = false;
    }
  }

  static void TearDownTestSuite() {
    if (npu_initialized_) {
      try {
        KernelRegistry::get_instance().cleanup();
        torch_npu::finalize_npu();
      } catch (...) {
      }
    }
  }

  void SetUp() override {
    npu_available_ = npu_initialized_;
    if (!npu_available_) {
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
      return;
    }
    tensor_options_ = torch::TensorOptions()
                          .dtype(torch::kBFloat16)
                          .device("npu:" + std::to_string(kDeviceId));
    torch::manual_seed(42);
    kernel_name_ = "rope_inplace_kernel";
    binary_filename_ = "rope_inplace_kernel.npubin";
    binary_path_ = GetKernelBinaryPath(binary_filename_);
  }

  void TearDown() override {}

  torch::TensorOptions tensor_options_;
  bool npu_available_ = false;
  std::string device_str_ = "npu:" + std::to_string(kDeviceId);
  std::string binary_filename_;
  std::string kernel_name_;
  std::string binary_path_;
};

bool TritonRopeInplaceTest::npu_initialized_ = false;

TEST_P(TritonRopeInplaceTest, RopeInplaceKernelTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  const int64_t batch = std::get<0>(GetParam());
  const int64_t head = std::get<1>(GetParam());
  const int64_t hidden_size = 512;
  const int64_t rope_dim = 64;

  ASSERT_GE(hidden_size, rope_dim);
  ASSERT_EQ(rope_dim % 2, 0);

  auto device = at::Device(device_str_);
  auto options_cpu_bf16 =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
  auto options_npu_bf16 =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);

  // x: (batch, head, hidden_size)；sin/cos: (batch, rope_dim）
  auto x_cpu = torch::randn({batch, head, hidden_size}, options_cpu_bf16);
  auto sin_cpu = torch::randn({batch, rope_dim}, options_cpu_bf16);
  auto cos_cpu = torch::randn({batch, rope_dim}, options_cpu_bf16);

  torch::Tensor out_golden = rope_ref_cpu(x_cpu, sin_cpu, cos_cpu, rope_dim);

  auto x_npu = x_cpu.to(device).contiguous();
  auto sin_npu = sin_cpu.to(device).contiguous();
  auto cos_npu = cos_cpu.to(device).contiguous();

  rope_inplace(x_npu, sin_npu, cos_npu, static_cast<uint32_t>(rope_dim));

  auto npu_stream = c10_npu::getCurrentNPUStream(kDeviceId);
  aclrtSynchronizeStream(npu_stream.stream());

  auto out_npu_cpu = x_npu.cpu().contiguous();
  auto diff = torch::abs(out_npu_cpu - out_golden);
  float max_diff = torch::max(diff).item<float>();
  EXPECT_LT(max_diff, kTolerance)
      << "rope_inplace batch=" << batch << " head=" << head << " max diff ("
      << max_diff << ") > tolerance (" << kTolerance << ")";
}

INSTANTIATE_TEST_SUITE_P(RopeInplaceParams,
                         TritonRopeInplaceTest,
                         ::testing::Values(std::make_tuple(1, 8),
                                           std::make_tuple(4, 8),
                                           std::make_tuple(8, 8),
                                           std::make_tuple(1, 1),
                                           std::make_tuple(4, 1),
                                           std::make_tuple(8, 1),
                                           std::make_tuple(16, 8)));

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
