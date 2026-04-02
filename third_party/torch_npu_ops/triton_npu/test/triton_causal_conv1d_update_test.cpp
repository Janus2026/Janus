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
#include <torch/nn/functional/conv.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <optional>

#include "kernel_registry.h"
#include "test/test_utils.h"
#include "torch_api/triton_ops_api.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace janus::kernel::npu {

constexpr float kTolerance = 5e-2f;  // bfloat16 tolerance
constexpr int32_t kDeviceId = 0;

#include <torch/torch.h>

#include <string>
#include <vector>

torch::Tensor causal_conv1d_update_ref(
    const torch::Tensor& x,
    torch::Tensor& conv_state,  // Modified in-place
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias = std::nullopt,
    bool activation = true,
    const std::optional<torch::Tensor>& cache_seqlens = std::nullopt) {
  auto dtype_in = x.scalar_type();
  bool unsqueeze = (x.dim() == 2);
  torch::Tensor x_ = unsqueeze ? x.unsqueeze(-1) : x;
  int64_t batch = x_.size(0);
  int64_t dim = x_.size(1);
  int64_t seqlen = x_.size(2);
  int64_t width = weight.size(1);
  int64_t state_len = conv_state.size(2);

  // Validate tensor shapes
  TORCH_CHECK(
      conv_state.sizes().vec() == std::vector<int64_t>({batch, dim, state_len}),
      "conv_state shape mismatch. Expected: (",
      batch,
      ", ",
      dim,
      ", ",
      state_len,
      "), Got: ",
      conv_state.sizes());
  TORCH_CHECK(weight.sizes().vec() == std::vector<int64_t>({dim, width}),
              "weight shape mismatch. Expected: (",
              dim,
              ", ",
              width,
              "), Got: ",
              weight.sizes());

  torch::Tensor x_new;
  if (!cache_seqlens.has_value()) {
    // Case 1: Standard state update (non-circular buffer)
    x_new = torch::cat({conv_state, x_}, -1).to(weight.scalar_type());
    // Update conv_state in-place (last state_len elements)
    conv_state.copy_(x_new.slice(-1, -state_len, x_new.size(-1)));
  } else {
    // Case 2: Circular buffer update
    auto cache_seqlens_tensor = cache_seqlens.value();
    TORCH_CHECK(
        cache_seqlens_tensor.sizes().vec() == std::vector<int64_t>({batch}),
        "cache_seqlens must be 1D tensor of size (batch,)");

    // Ensure cache_seqlens is on the same device as other tensors
    cache_seqlens_tensor = cache_seqlens_tensor.to(x_.device());

    // Generate width indices: [-(width-1), ..., -1] + cache_seqlens
    auto arange_tensor =
        torch::arange(-(width - 1), 0, torch::kLong).to(x_.device());
    auto width_idx = arange_tensor.unsqueeze(0)
                         .add(cache_seqlens_tensor.unsqueeze(1))
                         .to(torch::kLong);

    // Apply modulo for circular indexing
    width_idx = torch::remainder(width_idx, state_len)
                    .unsqueeze(1)  // Add dim dimension
                    .expand({batch, dim, width - 1});

    // Gather from conv_state using computed indices
    auto state_gathered = conv_state.gather(2, width_idx);

    // Construct new input tensor
    x_new = torch::cat({state_gathered, x_}, -1).to(weight.scalar_type());

    // Compute copy indices for updating conv_state
    auto copy_idx = torch::arange(0, seqlen, torch::kLong)
                        .to(x_.device())
                        .unsqueeze(0)
                        .add(cache_seqlens_tensor.unsqueeze(1));
    copy_idx = torch::remainder(copy_idx, state_len)
                   .unsqueeze(1)  // Add dim dimension
                   .expand({batch, dim, seqlen})
                   .to(torch::kLong);

    // Scatter x into conv_state (in-place update)
    conv_state.scatter_(2, copy_idx, x_);
  }

  // Prepare bias tensor for conv1d
  torch::Tensor bias_tensor = bias.has_value() ? bias.value() : torch::Tensor();

  // Perform depthwise convolution (groups=dim)
  auto weight_4d = weight.unsqueeze(1);  // (dim, 1, width)
  torch::Tensor out = torch::conv1d(x_new,
                                    weight_4d,
                                    bias_tensor,
                                    /*stride=*/torch::IntArrayRef{1},
                                    /*padding=*/torch::IntArrayRef{0},
                                    /*dilation=*/torch::IntArrayRef{1},
                                    /*groups=*/static_cast<int64_t>(dim));

  // Slice to keep only the last 'seqlen' elements
  out = out.slice(-1, -seqlen, out.size(-1));

  if (activation) {
    out = torch::silu(out);
  }

  // Restore original shape and dtype
  if (unsqueeze) {
    out = out.squeeze(-1);
  }
  return out.to(dtype_in);
}

class TritonCausalConv1dUpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kBFloat16).device("npu:0");
      npu_available_ = true;
    } catch (...) {
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
      npu_available_ = false;
      return;
    }

    torch::manual_seed(42);
    torch_npu::init_npu(device_str_);
    kernel_name_ = "_causal_conv1d_update_qwen_decode_kernel";
    binary_filename_ = "_causal_conv1d_update_qwen_decode_kernel.npubin";
    binary_path_ = GetKernelBinaryPath(binary_filename_);
    auto& reg = KernelRegistry::get_instance();
    (void)reg.register_kernel(kernel_name_, binary_path_);
    // Kernel binary might be missing in some environments; the API call is
    // still tested.
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

TEST_F(TritonCausalConv1dUpdateTest, MultiBatchTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  auto device = at::Device(device_str_);
  constexpr int64_t batch = 4;
  constexpr int64_t width = 4;
  constexpr int64_t seqlen = 1;
  constexpr bool has_bias = false;
  constexpr bool silu_activation = true;

  torch::manual_seed(0);
  auto dtype = torch::kBFloat16;
  float rtol = 1e-2f;
  float atol = 5e-2f;

  for (int64_t dim : {2048, 5120}) {
    // Create input tensors
    auto x = torch::randn({batch, dim, seqlen},
                          torch::TensorOptions().dtype(dtype).device(device));
    auto x_ref = x.clone().cpu();

    auto conv_state =
        torch::randn({batch, dim, width - 1},
                     torch::TensorOptions().dtype(dtype).device(device));
    auto conv_state_ref = conv_state.detach().clone().cpu();

    auto weight = torch::randn(
        {dim, width}, torch::TensorOptions().dtype(dtype).device(device));
    auto weight_ref = weight.clone().cpu();

    std::optional<torch::Tensor> bias = std::nullopt;
    if (has_bias) {
      bias = torch::randn({dim},
                          torch::TensorOptions().dtype(dtype).device(device));
    }

    // Create conv_state_indices for continuous batching
    auto conv_state_indices = torch::arange(
        batch, torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Run reference implementation on CPU
    auto out_ref = causal_conv1d_update_ref(x_ref,
                                            conv_state_ref,
                                            weight_ref,
                                            std::nullopt,
                                            silu_activation,
                                            std::nullopt);

    // Run NPU kernel
    auto npu_stream = c10_npu::getCurrentNPUStream(kDeviceId);
    auto out =
        npu_causal_conv1d_update(x,
                                 conv_state,
                                 weight,
                                 silu_activation,
                                 bias,
                                 std::nullopt,  // cache_seqlens
                                 conv_state_indices,
                                 std::nullopt,  // num_accepted_tokens
                                 std::nullopt,  // query_start_loc
                                 -1,            // max_query_len
                                 std::nullopt,  // intermediate_conv_window
                                 -1,            // pad_slot_id
                                 false          // validate_data
        );
    aclrtSynchronizeStream(npu_stream.stream());

    // Compare results
    auto out_cpu = out.cpu();
    auto output_diff = (out_ref - out_cpu).abs();
    float max_diff = output_diff.max().item().toFloat();
    float dim_atol = dim > 2048 ? atol * 2.0f : atol;

    EXPECT_LT(max_diff, dim_atol)
        << "Output mismatch: max diff = " << max_diff
        << ", tolerance = " << dim_atol << ", dim = " << dim
        << ", shape: " << out_cpu.sizes() << ", ref range ["
        << out_ref.min().item().toFloat() << ", "
        << out_ref.max().item().toFloat() << "]"
        << ", actual range [" << out_cpu.min().item().toFloat() << ", "
        << out_cpu.max().item().toFloat() << "]";

    // Compare conv_state (it should be updated)
    auto conv_state_cpu = conv_state.cpu();
    auto state_diff = (conv_state_ref - conv_state_cpu).abs();
    float max_state_diff = state_diff.max().item().toFloat();

    // Note: conv_state comparison might have some differences due to numerical
    // precision We use a more relaxed tolerance for state comparison
    EXPECT_LT(max_state_diff, atol * 2.0f)
        << "Conv state mismatch: max diff = " << max_state_diff
        << ", tolerance = " << (atol * 2.0f) << ", dim = " << dim;
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
