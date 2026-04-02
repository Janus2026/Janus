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

#include "operation_factory.h"
#include "triton_ops_api.h"

namespace janus::kernel::npu {

torch::Tensor npu_causal_conv1d_update(
    torch::Tensor& x,
    torch::Tensor& conv_state,
    torch::Tensor& weight,
    bool activation,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& cache_seqlens,
    const std::optional<torch::Tensor>& conv_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& query_start_loc,
    int32_t max_query_len,
    const std::optional<torch::Tensor>& intermediate_conv_window,
    int32_t pad_slot_id,
    bool validate_data) {
  (void)max_query_len;
  (void)validate_data;

  if (query_start_loc.has_value()) {
    LOG(ERROR)
        << "Current op does not support non-empty values for query_start_loc.";
    return torch::zeros(1);
  }
  bool unsqueeze = x.dim() == 2;
  if (unsqueeze) {
    x = x.unsqueeze(-1);
  }
  torch::Tensor out = torch::empty(
      x.sizes(), torch::TensorOptions().dtype(x.dtype()).device(x.device()));
  int32_t batch = x.size(0);
  int32_t dim = x.size(1);
  int32_t seqlen = x.size(2);
  int32_t width = weight.size(1);
  int32_t state_len = conv_state.size(2);

  auto npu_stream = c10_npu::getCurrentNPUStream();
  rtStream_t stream = static_cast<rtStream_t>(npu_stream.stream());

  const bool use_qwen_decode_kernel =
      !bias.has_value() && x.dim() == 3 && conv_state.dim() == 3 &&
      weight.dim() == 2 && x.size(1) == conv_state.size(1) &&
      x.size(1) == weight.size(0) && seqlen == 1 && width == 4 &&
      state_len == 3;
  if (use_qwen_decode_kernel) {
    auto x_contiguous = x.contiguous();
    auto state_contiguous = conv_state.contiguous();
    auto weight_contiguous = weight.contiguous();

    void* x_ptr = x_contiguous.data_ptr();
    void* conv_state_ptr = state_contiguous.data_ptr();
    void* weight_ptr = weight_contiguous.data_ptr();
    void* conv_state_indices_ptr = conv_state_indices.has_value()
                                       ? conv_state_indices.value().data_ptr()
                                       : nullptr;
    void* out_ptr = out.data_ptr();

    int32_t gridX = batch;
    int32_t gridY = (dim + 255) / 256;
    int32_t gridZ = 1;
    auto& op = OperationFactory::instance().causal_conv1d_update_qwen_decode();
    rtError_t ret =
        op.execute(stream, gridX, gridY, gridZ, [&](ArgsBuilder& ab) {
          ab.constructArgs(x_ptr,
                           conv_state_ptr,
                           weight_ptr,
                           conv_state_indices_ptr,
                           out_ptr,
                           pad_slot_id,
                           batch,
                           dim);
        });
    if (ret != RT_ERROR_NONE) {
      LOG(ERROR) << "rtKernelLaunch failed for "
                    "'_causal_conv1d_update_qwen_decode_kernel': "
                 << ret;
      return unsqueeze ? out.squeeze(-1) : out;
    }
    conv_state.copy_(state_contiguous);
    return unsqueeze ? out.squeeze(-1) : out;
  }

  void* x_ptr = x.data_ptr();
  void* conv_state_ptr = conv_state.data_ptr();
  void* weight_ptr = weight.data_ptr();
  void* conv_state_indices_ptr = conv_state_indices.has_value()
                                     ? conv_state_indices.value().data_ptr()
                                     : nullptr;
  void* out_ptr = out.data_ptr();

  int32_t gridX = 1, gridY = 1, gridZ = 1;
  rtError_t ret;
  if (!cache_seqlens.has_value() && !num_accepted_tokens.has_value()) {
    gridX = batch;
    auto& op =
        OperationFactory::instance().causal_conv1d_update_no_cache_no_mtp();
    ret = op.execute(stream, gridX, gridY, gridZ, [&](ArgsBuilder& ab) {
      ab.constructArgs(x_ptr,
                       conv_state_ptr,
                       weight_ptr,
                       conv_state_indices_ptr,
                       out_ptr,
                       pad_slot_id,
                       batch);
    });
    if (ret != RT_ERROR_NONE) {
      LOG(ERROR) << "rtKernelLaunch failed for "
                    "'_causal_conv1d_update_kernel_no_cache_len_no_mtp': "
                 << ret;
      return out;
    }
  } else {
    LOG(ERROR) << "currently causal_conv1d_update don't support input "
                  "cache_len and mtp scenerios.";
  }
  return out;
}

}  // namespace janus::kernel::npu
