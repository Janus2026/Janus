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

#ifndef ATB_SPEED_MODELS_COMMON_SELF_ATTENTION_H
#define ATB_SPEED_MODELS_COMMON_SELF_ATTENTION_H

#include <map>
#include <atb/atb_infer.h>
#include "operations/fusion/attention/fusion_attention.h"

namespace atb_speed {
namespace common {

/// This function adds kv cache movement operation and attention operations to the graph.
/// It supports flash attention and paged attention.
/// It supports ATB backend and AclNN backend.
///
/// \tparam NormParamType Types of the normalization parameters. Avaliable options are `atb::infer::RmsNormParam`
/// and `atb::infer::LayerNormParam`.
/// \param opGraph A reference to the graph
/// \param param Parameters for the FusionAttention module
/// \param operation the address of a pointer to a default operation
/// \param tensorMap Defines all the required tensors for the current graph, with the key representing
/// the input tensor name and the value corresponding to the tensor index.
/// Tensors are ordered by input tensors, output tensors and internal tensors.
/// \return A flag indicating whether the operation has been successfully created.
///
/// This function will use the following tensors:
/// Key in `tensorMap` | Requirements | Dtype            | Shape | Description |
/// -------------------|--------------|------------------|-------|----------|
/// in_seq_len         | Required     | int32 | [batch_size] | The total number of input and output tokens. <br> In the prefill phase, each elements equals to the length of the prompt. <br> For flash attention, each element is set to 1 in the decode phase. <br> For paged attention, each element is set to the number of input tokens plus output tokens in the decode phase. |
/// in_k_cache         | ^            | float16/bfloat16/int8 | [num_block,block_size,head_num,head_dim] | |
/// in_v_cache         | ^            | ^ | ^ | |
/// in_attention_mask  | ^            | Refer to SelfAttetion/PagedAttention Operation in the `atb/infer_op_params.h` and AttnOperation in the `operations/aclnn/ops/attn_operation.h` for more details. |||
/// in_token_offset    | ^            | int32 | [batch] | Token offset after calculation. Used only if `isFA` is true. |
/// in_layer_id        | ^            | int32 | [1] | The index of kv cache for the current layer. Used only if `isFA` is true.  |
/// in_block_tables    | ^            | int32 | [num_tokens, max_num_blocks_per_query] | Used only if `isFA` is false. |
/// in_slots_in_pa_or_logn_in_fa | ^  | float32 | `isFA` is true (Prefill phase): [maxSeqLen] | Logarithmic scaling. |
/// ^                  | ^            | ^       | `isFA` is true (Decode phase): [batch_size] | ^ |
/// ^                  | ^            | ^       | `isFA` is false: [num_tokens] | Storage offset of each token key or value in the cache. |
/// in_slopes          | `param.selfAttentionParam.maskType` is in one of `atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS`, `atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT` and `atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN` | Atlas 800I A2: float32; Atlas 300I DUO: float16  | [head_num] | It is the coefficient of each head of the alibi mask. |
/// in_batch_wins      | `param.pageAttentionParam.compressType` equals to `atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD` or `atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE` | int32 | [batch*head_num] | Compressed window size |
/// in_ra_seq_len      | ^            | int32   | [batch_size] | |
/// in_pffset_index    | `param.pageAttentionParam.compressType == atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE` | int32 | [batch*head_num] | |
/// in_ra_offset       | ^            | float32 | [num_blocks,block_size] | |
/// in_reshape_seq_len | ^            | int32   | [batch] | |
/// in_q_len           | `param.pageAttentionParam.calcType == atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC` | int32 | [batch] | Number of input tokens for the current forward pass. |
/// in_k_dequant_scale | `param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION` | Refer to the table below for further details. |||
/// in_v_dequant_scale | ^            | ^ |||
/// in_k_dequant_offset| `param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION` and `param.pageAttentionParam.hasQuantOffset` is true | Refer to the table below for further details. |||
/// in_v_dequant_offset| ^            | ^ |||
/// in_log_n_scale     | `param.pageAttentionParam.scaleType == atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_LOGN` | float32 | [batch] | logarithmic scaling |
/// in_qk_descale      | `param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE` | float32 | [head_num] | |
/// fa3_v_quant_scale  | ^            | float32 | [head_num] | |
/// intermediate_q     | Required     | float16/bfloat16 | `isFA` is true: [bsz,seq_len,head_num,head_dim]      | The output of the q linear operation. |
/// ^                  | ^            | ^                | `isFA` is false: [len(all_seq),head_num,head_dim]    | ^                                     |
/// intermediate_k     | ^            | ^                | `isFA` is true: [bsz,seq_len,kv_head_num,head_dim]   | The output of the k linear operation. |
/// ^                  | ^            | ^                | `isFA` is false: [len(all_seq),kv_head_num,head_dim] | ^                                     |
/// intermediate_v     | ^            | ^                | `isFA` is true: [bsz,seq_len,kv_head_num,head_dim]   | The output of the v linear operation. |
/// ^                  | ^            | ^                | `isFA` is false: [len(all_seq),kv_head_num,head_dim] | ^                                     |
/// intermediate_self_attention | ^   | ^ | `isFA` is true: [bsz,seq_len,head_num*head_dim]   | The output of the attention operation. |
/// ^                           | ^   | ^ | `isFA` is false: [len(all_seq),head_num*head_dim] | ^ |
/// intermediate_k_int8 | `param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION` | int8 | The same as intermediate_k | intermediate_k after int8 quantization |
/// intermediate_v_int8 | ^ | int8 | The same as intermediate_k | intermediate_v after int8 quantization |
/// intermediate_q_int8 | `param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION` | int8 | The same as intermediate_k | intermediate_q after int8 quantization |
/// ^                   | `!param.isPrefill && param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE` | ^ | ^ | ^ |
///
/// Detailed information for the in_k_dequant_scale, in_v_dequant_scale, in_k_dequant_offset and in_v_dequant_offset.
/// Key in `tensorMap` | Dtype of the output tenosr | Attention Backend | Dtype | Shape | Description |
/// -------------------|----------------------------|-------------------|-------|-------|----------|
/// in_k_dequant_scale | float16                    | ATB               | int64 | [head_num*head_dim] | |
/// in_v_dequant_scale | ^                          | ^                 | ^     | ^     | |
/// in_k_dequant_offset| ^                          | ^                 | int32 | ^     | |
/// in_v_dequant_offset| ^                          | ^                 | ^     | ^     | |
/// in_k_dequant_scale | bfloat16                   | ATB               | float32 | ^   | |
/// in_v_dequant_scale | ^                          | ^                 | ^     | ^     | |
/// in_k_dequant_offset| ^                          | ^                 | int32 | ^     | |
/// in_v_dequant_offset| ^                          | ^                 | ^     | ^     | |
/// in_k_dequant_scale | float16                    | AclNN             | float16 | [2,head_num*head_dim] | |
/// in_v_dequant_scale | ^                          | ^                 | ^     | [1]   | placeholder|
/// in_k_dequant_offset| ^                          | ^                 | ^     | [2,head_num*head_dim]   | placeholder |
/// in_v_dequant_offset| ^                          | ^                 | ^     | [1]   | placeholder |
/// in_k_dequant_scale | bfloat16                   | AclNN             | bfloat16 | [2,head_num*head_dim] | placeholder |
/// in_v_dequant_scale | ^                          | ^                 | ^     | [1]   | placeholder |
/// in_k_dequant_offset| ^                          | ^                 | ^     | [2,head_num*head_dim] | placeholder |
/// in_v_dequant_offset| ^                          | ^                 | ^     | [1]   | placeholder |
///
/// Example:
/// \code
/// atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
/// // Modify fusionAttentionParam's attribute if needed.
/// // Define all the required tensors and corresponding tensor index.
/// std::map<std::string, uint32_t> tensorMap = {{"in_k_cache", 0}, {"in_v_cache", 1}, ...}
/// atb::GraphParam opGraph;
/// AddSelfAttention(opGraph, fusionAttentionParam, tensorMap);
/// \endcode
template <typename NormParamType>
int64_t AddSelfAttention(
    atb::GraphParam &opGraph, const FusionAttentionParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap);
    
} // namespace common
} // namespace atb_speed
#endif