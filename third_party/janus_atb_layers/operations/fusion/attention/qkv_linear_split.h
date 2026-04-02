/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ATB_SPEED_MODELS_COMMON_QKV_H
#define ATB_SPEED_MODELS_COMMON_QKV_H

#include <atb/atb_infer.h>
#include "operations/fusion/attention/fusion_attention.h"


namespace atb_speed {
namespace common {

/// This function performs normalization and qkv linear operations.
/// It supports grouped query attention, multi head attention and quantization scenarios.
/// It also accepts packed qkv linear weights.
///
/// \tparam NormParamType Types of the normalization parameters. Avaliable options are `atb::infer::RmsNormParam`
/// and `atb::infer::LayerNormParam`.
/// \param param Parameters for the FusionAttention module
/// \param operation the address of a pointer to a default operation
/// \return A flag indicating whether the operation has been successfully created.
///
/// Operation's inputs:
/// Name                   | Requirements | Dtype            | Shape | Description |
/// -----------------------|--------------|------------------|-------|----------|
/// in_qkv_input           | Required     | float16/bfloat16 | `isFA` is false: [len(all_seq),num_heads,head_dim] | Hidden States |
/// ^                      | ^            | ^                | `isFA` is true: [bsz,seq_len,num_heads,head_dim]   | ^ |
/// in_qkv_norm_weight     | ^            | Refer to `atb_speed::common::NormLinear` in the `operations/fusion/norm/norm_linear.h` for more details. |||
/// in_qkv_norm_bias       | ^            | ^ |||
/// in_qkv_norm_new_weight | ^            | ^ |||
/// in_qkv_norm_new_bias   | ^            | ^ |||
/// in_qkv_weight_0        | ^            | If qkv are packed, these are concatenated qkv linear weights; otherwise, these are weights for the q linear operation. <br> Refer to `atb_speed::common::FusionLinear` in the `operations/fusion/linear/linear.h` for more details. |||
/// in_qkv_scale_0         | ^            | ^ |||
/// in_qkv_offset_0        | ^            | ^ |||
/// in_qkv_descale_0       | ^            | ^ |||
/// in_qkv_bias_0          | ^            | ^ |||
/// in_qkv_compress_idx_0  | ^            | ^ |||
/// in_qkv_weight_1        | ^            | If qkv are not packed, these are weights for the k linear operation; otherwise, placeholders should be provided. <br> Refer to `atb_speed::common::FusionLinear` in the `operations/fusion/linear/linear.h` for more details. |||
/// in_qkv_scale_1         | ^            | ^ |||
/// in_qkv_offset_1        | ^            | ^ |||
/// in_qkv_descale_1       | ^            | ^ |||
/// in_qkv_bias_1          | ^            | ^ |||
/// in_qkv_compress_idx_1  | ^            | ^ |||
/// in_qkv_weight_2        | ^            | If qkv are not packed, these are weights for the v linear operation; otherwise, placeholders should be provided. <br> Refer to `atb_speed::common::FusionLinear` in the `operations/fusion/linear/linear.h` for more details. |||
/// in_qkv_scale_2         | ^            | ^ |||
/// in_qkv_offset_2        | ^            | ^ |||
/// in_qkv_descale_2       | ^            | ^ |||
/// in_qkv_bias_2          | ^            | ^ |||
/// in_qkv_compress_idx_2  | ^            | ^ |||
/// in_im_mask             | `param.supportLora` is true and `param.useImMask` is true | Refer to `atb_speed::common::FusionLinear` in the `operations/fusion/linear/linear.h` for more details. |||
/// in_seq_len_cum_sum     | `param.supportLora` is true | ^ |||
/// in_qkv_lora_a_0        | ^            | ^ |||
/// in_qkv_lora_b_0        | ^            | ^ |||
/// in_qkv_lora_a_1        | ^            | ^ |||
/// in_qkv_lora_b_1        | ^            | ^ |||
/// in_qkv_lora_a_2        | ^            | ^ |||
/// in_qkv_lora_b_2        | ^            | ^ |||
/// in_q_norm_weight       | `param.useQKNorm` is true | | | |
/// in_k_norm_weight       | `param.useQKNorm` is true | | | |
///
/// Operations's outputs:
/// Name       | Dtype                    | Shape                                                |
/// -----------|--------------------------|------------------------------------------------------|
/// out_q      | The same as in_qkv_input | `isFA` is false: [len(all_seq),head_num,head_dim]    |
/// ^          | ^                        | `isFA` is true: [bsz,seq_len,head_num,head_dim]      |
/// out_k      | The same as in_qkv_input | `isFA` is false: [len(all_seq),kv_head_num,head_dim] |
/// ^          | ^                        | `isFA` is true: [bsz,seq_len,kv_head_num,head_dim]   |
/// out_v      | The same as in_qkv_input | `isFA` is false: [len(all_seq),kv_head_num,head_dim] |
/// ^          | ^                        | `isFA` is true: [bsz,seq_len,kv_head_num,head_dim]   |
///
/// Example:
/// \code
/// atb::Node qkvLinearSplitNode;
/// atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
/// // Modify fusionAttentionParam's attribute if needed.
/// QKVLinearSplit(fusionAttentionParam, &qkvLinearSplitNode.operation);
/// qkvLinearSplitNode.inTensorIds = {...};  // Passing inputs for the operation in order
/// qkvLinearSplitNode.outTensorIds = {...};  // Tensor index for out_q, out_k, out_v
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(qkvLinearSplitNode);
/// \endcode
template <typename NormParamType>
atb::Status QKVLinearSplit(const FusionAttentionParam<NormParamType> &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif