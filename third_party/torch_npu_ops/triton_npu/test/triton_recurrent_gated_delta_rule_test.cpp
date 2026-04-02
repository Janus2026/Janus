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

#include <optional>

#include "kernel_registry.h"
#include "test/test_utils.h"
#include "torch_api/triton_ops_api.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace janus::kernel::npu {

constexpr float kOutputRtol = 0.005f;
constexpr float kOutputAtol = 0.01f;
constexpr float kStateRtol = 0.05f;
constexpr float kStateAtol = 0.01f;
constexpr int32_t kDeviceId = 0;

torch::Tensor l2norm(const torch::Tensor& x,
                     int64_t dim = -1,
                     float eps = 1e-6f) {
  auto inv_norm = torch::rsqrt((x * x).sum(dim, /*keepdim=*/true) + eps);
  return x * inv_norm;
}

// Direct port of Python recurrent_gated_delta_rule_ref (lines 385-419).
// Caller must pass pre-normalized q,k when use_qk_l2norm_in_kernel is true.
std::pair<torch::Tensor, torch::Tensor> torch_recurrent_gated_delta_rule(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    const std::optional<torch::Tensor>& initial_state,
    bool output_final_state,
    const std::optional<float>& scale_opt) {
  auto initial_dtype = query.scalar_type();
  auto device = query.device();

  // transpose(1,2): (B,T,H,K) -> (B,H,T,K), same as Python ref
  auto query_ = query.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto key_ = key.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto value_ = value.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto g_ = g.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto beta_ = beta.transpose(1, 2).contiguous().to(torch::kFloat32);

  int64_t batch_size = key_.size(0);
  int64_t num_heads = key_.size(1);
  int64_t sequence_length = key_.size(2);
  int64_t k_head_dim = key_.size(3);
  int64_t v_head_dim = value_.size(3);

  // scale: default 1/sqrt(K) as in Python ref
  float scale = scale_opt.has_value()
                    ? scale_opt.value()
                    : 1.0f / std::sqrt(static_cast<float>(k_head_dim));
  query_ = query_ * scale;

  auto core_attn_out = torch::zeros(
      {batch_size, num_heads, sequence_length, v_head_dim}, value_.options());
  torch::Tensor h;

  if (initial_state.has_value() && initial_state.value().defined()) {
    h = initial_state.value().to(torch::kFloat32).to(device).contiguous();
  } else {
    h = torch::zeros({batch_size, num_heads, k_head_dim, v_head_dim},
                     value_.options());
  }

  for (int64_t i = 0; i < sequence_length; ++i) {
    auto b_q = query_.select(2, i);
    auto b_k = key_.select(2, i);
    auto b_v = value_.select(2, i).clone();
    auto b_g = g_.select(2, i).exp().unsqueeze(-1).unsqueeze(-1);
    auto b_beta = beta_.select(2, i).unsqueeze(-1);

    // h = h.clone() * g.exp()[..., None, None]
    h = h.clone() * b_g;
    // b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
    b_v = b_v - (h.clone() * b_k.unsqueeze(-1)).sum(-2);
    // b_v = b_v * b_beta[..., None]
    b_v = b_v * b_beta;
    // h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
    h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2);
    // o[:,:,i] = einsum("bhd,bhdm->bhm", b_q, h)
    core_attn_out.slice(2, i, i + 1) =
        (b_q.unsqueeze(-1) * h).sum(-2).unsqueeze(2);
  }

  if (!output_final_state) {
    h = torch::Tensor();
  }

  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_pair(core_attn_out, h);
}

class TritonRecurrentGatedDeltaRuleTest : public ::testing::Test {
 protected:
  static bool npu_initialized_;

  static void SetUpTestSuite() {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      torch_npu::init_npu("npu:" + std::to_string(kDeviceId));
      auto& reg = KernelRegistry::get_instance();
      std::string binary_path = GetKernelBinaryPath(
          "fused_recurrent_gated_delta_rule_fwd_kernel.npubin");
      npu_initialized_ =
          reg.register_kernel("fused_recurrent_gated_delta_rule_fwd_kernel",
                              binary_path) &&
          reg.get_kernel_stub("fused_recurrent_gated_delta_rule_fwd_kernel") !=
              nullptr;
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
          torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
      return;
    }
    tensor_options_ = torch::TensorOptions()
                          .dtype(torch::kFloat16)
                          .device("npu:" + std::to_string(kDeviceId));
    torch::manual_seed(42);
    kernel_name_ = "fused_recurrent_gated_delta_rule_fwd_kernel";
    binary_filename_ = "fused_recurrent_gated_delta_rule_fwd_kernel.npubin";
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

bool TritonRecurrentGatedDeltaRuleTest::npu_initialized_ = false;

// Test params aligned with Python test_accuracy_fused_recurrent:
// (B, T, H, HV, D, scale, gate_logit_normalizer, dtype)
struct RecurrentTestParam {
  int64_t B, T, H, HV, D;
  float scale, gate_logit_normalizer;
  torch::ScalarType dtype;
};

class TritonRecurrentGatedDeltaRuleParamTest
    : public TritonRecurrentGatedDeltaRuleTest,
      public ::testing::WithParamInterface<RecurrentTestParam> {};

TEST_P(TritonRecurrentGatedDeltaRuleParamTest, AccuracyMatch) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  const auto& p = GetParam();
  auto device = at::Device(device_str_);
  const int64_t batch = p.B;
  const int64_t T = p.T;
  const int64_t num_heads = p.H;
  const int64_t num_v_heads = p.HV;
  const int64_t k_head_dim = p.D;
  const int64_t v_head_dim = p.D;
  const float scale_val = p.scale;
  const float gate_logit_normalizer = p.gate_logit_normalizer;
  constexpr bool use_qk_l2norm_in_kernel = true;

  torch::manual_seed(42);
  auto dtype = p.dtype;
  auto L = batch * T;

  // Input generation aligned with Python: beta.sigmoid(), g =
  // logsigmoid(rand)/gate_logit_normalizer
  auto q = torch::randn({batch, T, num_heads, k_head_dim}, dtype);
  auto k = torch::randn({batch, T, num_heads, k_head_dim}, dtype);
  auto v = torch::randn({batch, T, num_v_heads, v_head_dim}, dtype);
  auto beta = torch::rand({batch, T, num_v_heads}, dtype).sigmoid();
  auto g = torch::log_sigmoid(
               torch::rand({batch, T, num_v_heads}, torch::kFloat32)) /
           gate_logit_normalizer;
  auto initial_state = torch::randn(
      {batch, num_v_heads, k_head_dim, v_head_dim}, torch::kFloat32);

  torch::Tensor q_expanded = q, k_expanded = k;
  if (num_v_heads / num_heads > 1) {
    q_expanded = q.repeat_interleave(num_v_heads / num_heads, 2);
    k_expanded = k.repeat_interleave(num_v_heads / num_heads, 2);
  }

  // receives pre-normalized q,k; apply l2norm before calling
  if (use_qk_l2norm_in_kernel) {
    q_expanded = l2norm(q_expanded, -1, 1e-6f);
    k_expanded = l2norm(k_expanded, -1, 1e-6f);
  }

  auto [golden_o, golden_state] =
      torch_recurrent_gated_delta_rule(q_expanded,
                                       k_expanded,
                                       v,
                                       g,
                                       beta,
                                       initial_state,
                                       true,
                                       std::optional<float>(scale_val));

  auto q_d = q.reshape({1, L, num_heads, k_head_dim}).to(device);
  auto k_d = k.reshape({1, L, num_heads, k_head_dim}).to(device);
  auto v_d = v.reshape({1, L, num_v_heads, v_head_dim}).to(device);
  auto g_d = g.reshape({1, L, num_v_heads}).to(device);
  auto beta_d = beta.reshape({1, L, num_v_heads}).to(device);
  auto init_d = initial_state.clone().to(device);

  std::vector<int64_t> culen;
  culen.reserve(batch + 1);
  for (int i = 0; i <= batch; ++i) {
    culen.push_back(i * T);
  }
  auto cu_seqlens = torch::tensor(culen, torch::kInt64).to(device);
  auto ssm_state_indices = torch::arange(
      batch, torch::TensorOptions().dtype(torch::kInt32).device(device));

  auto npu_stream = c10_npu::getCurrentNPUStream(kDeviceId);
  auto [o_d, state_d] =
      npu_fused_recurrent_gated_delta_rule(q_d,
                                           k_d,
                                           v_d,
                                           g_d,
                                           beta_d,
                                           scale_val,
                                           init_d,
                                           true,
                                           cu_seqlens,
                                           ssm_state_indices,
                                           std::nullopt,
                                           use_qk_l2norm_in_kernel);
  aclrtSynchronizeStream(npu_stream.stream());

  auto o = o_d.cpu().reshape(golden_o.sizes());
  auto state = state_d.cpu();

  // rtol=0.005, atol=0.01 for output
  auto output_diff =
      (golden_o.to(torch::kFloat32) - o.to(torch::kFloat32)).abs();
  auto output_max_diff = output_diff.max().item<float>();
  EXPECT_LT(output_max_diff, kOutputAtol)
      << "Output mismatch: max diff = " << output_max_diff
      << ", shape: " << o.sizes();

  // rtol=0.05, atol=0.01 for state
  auto state_diff =
      (golden_state.to(torch::kFloat32) - state.to(torch::kFloat32)).abs();
  auto state_max_diff = state_diff.max().item<float>();
  EXPECT_LT(state_max_diff, kStateAtol)
      << "State mismatch: max diff = " << state_max_diff
      << ", shape: " << state.sizes();
}

INSTANTIATE_TEST_SUITE_P(
    RecurrentParams,
    TritonRecurrentGatedDeltaRuleParamTest,
    ::testing::Values(
        RecurrentTestParam{1, 1, 4, 8, 128, 0.5f, 0.1f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 4, 8, 128, 0.5f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{4, 1, 4, 8, 128, 0.2f, 0.1f, torch::kBFloat16},
        RecurrentTestParam{8, 1, 4, 8, 128, 0.3f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{16, 1, 4, 8, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{32, 1, 4, 8, 128, 0.2f, 1.0f, torch::kBFloat16}));

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
