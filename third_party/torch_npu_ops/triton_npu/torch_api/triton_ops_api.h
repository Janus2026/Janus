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

#pragma once

#include <torch/torch.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace janus::kernel::npu {

void rope_inplace(torch::Tensor& x,
                  torch::Tensor& sin,
                  torch::Tensor& cos,
                  uint32_t rope_dim = 64);

std::pair<torch::Tensor, torch::Tensor> npu_fused_gdn_gating(
    torch::Tensor& A_log,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& dt_bias,
    float beta = 1.0f,
    float threshold = 20.0f);

std::pair<torch::Tensor, torch::Tensor> npu_fused_recurrent_gated_delta_rule(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& g,
    const std::optional<torch::Tensor>& beta = std::nullopt,
    const std::optional<float>& scale = std::nullopt,
    const std::optional<torch::Tensor>& initial_state = std::nullopt,
    bool inplace_final_state = true,
    const std::optional<torch::Tensor>& cu_seqlens = std::nullopt,
    const std::optional<torch::Tensor>& ssm_state_indices = std::nullopt,
    const std::optional<torch::Tensor>& num_accepted_tokens = std::nullopt,
    bool use_qk_l2norm_in_kernel = false);

torch::Tensor layer_norm_fwd(
    torch::Tensor& x,
    torch::Tensor& weight,
    torch::Tensor& bias,
    double eps,
    const std::optional<torch::Tensor>& z = std::nullopt,
    int64_t group_size = -1,
    bool norm_before_gate = true,
    bool is_rms_norm = false);

torch::Tensor npu_causal_conv1d_update(
    torch::Tensor& x,
    torch::Tensor& conv_state,
    torch::Tensor& weight,
    bool activation = true,
    const std::optional<torch::Tensor>& bias = std::nullopt,
    const std::optional<torch::Tensor>& cache_seqlens = std::nullopt,
    const std::optional<torch::Tensor>& conv_state_indices = std::nullopt,
    const std::optional<torch::Tensor>& num_accepted_tokens = std::nullopt,
    const std::optional<torch::Tensor>& query_start_loc = std::nullopt,
    int32_t max_query_len = -1,
    const std::optional<torch::Tensor>& intermediate_conv_window = std::nullopt,
    int32_t pad_slot_id = -1,
    bool validate_data = false);

}  // namespace janus::kernel::npu
