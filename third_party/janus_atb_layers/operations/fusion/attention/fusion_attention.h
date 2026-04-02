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

#ifndef ATB_SPEED_MODELS_COMMON_ATTENTION_H
#define ATB_SPEED_MODELS_COMMON_ATTENTION_H

#include <vector>
#include <map>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "operations/aclnn/ops/attn_operation.h"

namespace atb_speed {
namespace common {

/// The categories of the FusionAttention's input tensors
/// Input tensors will be arragned according to the order of their categories
enum AttnInTensorCategory : unsigned int {
    /// Default tensors
    ATTN_DEFAULT = 0,
    /// Tensors required by addRmsNormQuant, addRmsNormDynamicQuant
    ATTN_ADD_RMS_NORM_QUANT,
    /// Tensors required when passing compressed alibi mask
    ATTN_ALIBI_MASK_COMPRESS,
    /// Tensors required by kv head compression with alibi mask
    ATTN_COMPRESS_HEAD_ALIBI,
    /// Tensors required by kv head compression with rope
    ATTN_COMPRESS_HEAD_ROPE,
     ATTN_OMNI,
    /// Tensors required by speculation
    ATTN_SPECULATE,
    /// Tensors required by int8 quantization for the KV cache
    ATTN_KV_QUANT_SCALE,
    /// Tensors required by int8 quantization for the KV cache
    ATTN_KV_QUANT_OFFSET,
    /// Tensors required by flash attention 3
    ATTN_FA3,
    /// The mask tensor before applying lora adapters
    ATTN_LORA_MASK,
    /// Tensors needed for LoRA
    ATTN_LORA,
    /// Tensors required by the quantization of the all reduce operation
    ATTN_REDUCE_QUANT,
    /// Tensors required when applying logarithmic scaling to the attention
    ATTN_LOG_N_SCALE,
    ATTN_QK_NORM,
    /// Tensors required by add rmsnorm
    ATTN_ADD_NORM,
    /// Tensors required by CMO
    ATTN_CMO,
    /// Tensosr required by flashcomm1.0
    ATTN_FC,
    /// Tensors required by x attention
    ATTN_X_ATTENTION,
    /// Tensors required by acl graph
    ATTN_ACL_GRAPH,
    /// A flag signifying the end of all categories
    ATTN_END
};

/// The index of the q linear within the layer
const uint64_t Q_LINEAR_INDEX = 0;
/// The index of the k linear within the layer
const uint64_t K_LINEAR_INDEX = 1;
/// The index of the v linear within the layer
const uint64_t V_LINEAR_INDEX = 2;
/// The index of the dense linear within the layer
const uint64_t DENSE_LINEAR_INDEX = 3;

/// Parameters for the FusionAttention module
/// \tparam Types of the normalization parameters. Avaliable options are `atb::infer::RmsNormParam`
/// and ``atb::infer::LayerNormParam`.
template <typename NormParamType>
struct FusionAttentionParam {
    // QKV linear param
    /// A flag indicating whether the model structure is grouped query attention or multi head attention
    bool isGroupedQueryAttention = false;
    /// When `isBF16` is true, bfloat16 precision is used; otherwise, float16 precision is used.
    bool isBF16 = false;
    /// A flag indicating whether to reshape before spliting the packed output of the qkv linear operation
    bool splitWithStride = false;
    /// A flag indicating whether qkv linear has bias
    bool qkvHasBias = false;
    /// A flag indicating whether normalization is skipped
    bool skipNorm = false;
    /// A flag indicating whether normalization has bias
    bool normHasBias = false;
    /// A flag indicating whether to use NormQuant fusion operation
    bool enableNormQuantOp = true;
    /// A flag indecating whether to prefetch weight
    bool enablePreFetchWeight = false;
     /// A flag indicating whether lora is enabled.
    bool supportLora = false;
    /// A flag indicating whether a mask is used before applying lora adapter.
    bool useImMask = false;
    /// it should be activated when batch inputs include multiple LoRA adapters
    bool loraEnableGMM = false;
    /// A flag indicating whether using qnorm and knorm.
    bool useQKNorm = false;
    /// A flag indicating which norm type used by qnorm and knorm.
    bool rmsnormQKNorm = false;
    /// A flag indicating whether RopeQuantKvcache is enabled.
    bool enableRopeQuantKvcache = false;
    /// The backend of the attention module; refer to `OpBackend` for the supported values
    int attnBackend = atb_speed::common::OpBackend::ATB;
    /// The group size used for dequantizing the weight tensor in the per-group quantization approach
    int quantGroupSize = 0;
    /// Indicates the pack type and the quantization type of the qkv linear.
    int packQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    /// Specifies the quantization type for the following linear module:
    /// q linear, k linear, v linear, dense linear, gate linear, up linear, and down linear.
    std::vector<int> layerLinearQuantType = {};
    /// Specifies the weight description of the following linear module:
    /// qkv linear, dense linear, gateup linear and down linear.
    std::vector<int> layerLinearDescs = {
        common::LinearDesc::INVALID_DESC, common::LinearDesc::INVALID_DESC, common::LinearDesc::INVALID_DESC,
        common::LinearDesc::INVALID_DESC, common::LinearDesc::INVALID_DESC, common::LinearDesc::INVALID_DESC,
        common::LinearDesc::INVALID_DESC
    };
    /// Defines the transpose type of the second matrix in the matmul operation for the following linear module:
    /// q linear, k linear, v linear, dense linear, gate linear, up linear, and down linear.
    std::vector<int> layerLinearTransposeType = {};
    /// Normalization parameters for float operation
    NormParamType normParamType;
    /// Normlization parameters for quantization operation
    NormParamType normQuantParamType;
    // rope param
    /// The type of rotary position embedding. Refer to `RotaryType`
    /// in the `operations/fusion/positional_embedding.h` for more details.
    atb_speed::common::RotaryType rotaryType;
    /// Parameters for the rope operation
    atb::infer::RopeParam ropeParam;
    // self attention param
    /// A flag indicating whether to apply logarithmic scaling to the attention
    bool enableLogN = false;
    bool enableQScale = false;
    /// A flag indicating whether split fuse is enabled
    bool enableSplitFuse = false;
    /// If `isFA` is true, Flash Attention is used; otherwise, Paged Attention is used
    bool isFA = true;
    /// A flag indicating the prefill and decode phases
    bool isPrefill = false;
    /// The dimension per attention head
    int headDim = 0;
    /// Parameters for the self attention operation from the ATB backend
    atb::infer::SelfAttentionParam selfAttentionParam;
    /// Parameters for the page attention operation from the ATB backend
    atb::infer::PagedAttentionParam pageAttentionParam;
    /// Parameters for the reshape and cache operation from the ATB backend
    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    atb::infer::ReshapeAndCacheOmniParam  reshapeCacheOmniParm;
    /// Parameters for the attention operation from the AclNN backend (used in the decode phase)
    atb_speed::common::AclNNAttnParam aclnnIncreAttentionParam;
    // self out linear param
    /// A flag indicating whether dense linear has bias
    bool selfAttnHasBias = false;
    /// A flag that indicates whether low-latency computation over communication is enabled
    bool supportLcoc = false;
    bool enableMC2 = false;
    /// The quantization type of the dense linear. Refer to `PackQuantType` in the `operations/utils.h`.
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    /// Details about tensor parallelism
    atb_speed::common::TensorParallelInfo selfOutLinearTensorParallelInfo;
    /// A flag indicating whether to use the atb matmul backend
    int matmulBackend = atb_speed::common::OpBackend::ATB;
    /// A flag indicating whether addNorm fusion is enabled in attention
    bool enableAddNorm = false;
    /// Specifies whether the input norm enables antioutlier
    bool isAntiOutlier = false;
    /// Specifies whether enabled omni
    bool enableOmniattention = false;
    bool isomnicompressed = false;
    // Use Aclnn RmsNorm instead of ATB RmsNorm.
    bool enableAclnnRmsNorm = false;
    // enable PrefixCache without ChunkedPrefill.
    bool isPrefixCacheWithoutChunk = false;
    // A flag indicating whether use flashcomm1.0
    bool enableFlashComm = false;
    /// A flag indicating whether to use pmcc obfuscation
    bool enableModelConfuscation = false;
    /// A handle used by pmcc model obfuscation
    int32_t modelConfuscationFd = 0;
    /// Hidden size per rank
    int32_t hiddenSizePerRank = 0;
    /// A flag indicating whether to use the unified decode kv cache pipeline
    /// (prefill writes kv cache + decode consumes decode kv cache).
    bool enableXattention = false;
    /// The layer id
    int32_t layerId = 0;
    // enable acl graph
    bool enableAclGraphPagedAttention = false;
};

template <typename NormParamType>
std::map<std::string, uint32_t> ConstructTensorMap(const FusionAttentionParam<NormParamType> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);

/// This function is the main entrance for the fusion attention module.
/// It consists of QKVLinearSplit operation, Rope operation, Elementwise operation to quant intermediate kv tensors,
/// ReshapeAndCache operation, SelfAttention/PageAttention operation
/// and LinearParallel operation for the dense linear.
///
/// \tparam NormParamType Types of the normalization parameters. Avaliable options are `atb::infer::RmsNormParam`
/// and `atb::infer::LayerNormParam`.
/// \param param Parameters for the normalization and linear module
/// \param operation the address of a pointer to a default operation
/// \return A flag indicating whether the operation has been successfully created.
///
/// Operation's inputs:
/// Name               | Requirements | Dtype            | Shape | Description |
/// -------------------|--------------|------------------|-------|----------|
/// in_input           | Required     | float16/bfloat16 | `isFA` is false: [len(all_seq),num_heads,head_dim] | Hidden States |
/// ^                  | ^            | ^                | `isFA` is true: [bsz,seq_len,num_heads,head_dim]  | ^ |
/// in_norm_weight     | ^            | Refer to `atb_speed::common::QKVLinearSplit` in the `operations/fusion/attention/qkv_linear_split.h` for more details. |||
/// in_norm_bias       | ^            | ^ |||
/// in_norm_new_weight | ^            | ^ |||
/// in_norm_new_bias   | ^            | ^ |||
/// in_weight_0        | ^            | ^ |||
/// in_scale_0         | ^            | ^ |||
/// in_offset_0        | ^            | ^ |||
/// in_descale_0       | ^            | ^ |||
/// in_bias_0          | ^            | ^ |||
/// in_compress_idx_0  | ^            | ^ |||
/// in_weight_1        | ^            | ^ |||
/// in_scale_1         | ^            | ^ |||
/// in_offset_1        | ^            | ^ |||
/// in_descale_1       | ^            | ^ |||
/// in_bias_1          | ^            | ^ |||
/// in_compress_idx_1  | ^            | ^ |||
/// in_weight_2        | ^            | ^ |||
/// in_scale_2         | ^            | ^ |||
/// in_offset_2        | ^            | ^ |||
/// in_descale_2       | ^            | ^ |||
/// in_bias_2          | ^            | ^ |||
/// in_compress_idx_2  | ^            | ^ |||
/// in_cos_embed       | ^            | float16/bfloat16 | [len(all_seq),head_dim] | The cosine part of the rotary embedding. |
/// in_sin_embed       | ^            | ^                | ^                       | The sine part of the rotary embedding. |
/// in_seq_len         | ^            | int32 | [batch_size] | The total number of input and output tokens. <br> In the prefill phase, each elements equals to the length of the prompt. <br> For flash attention, each element is set to 1 in the decode phase. <br> For paged attention, each element is set to the number of input tokens plus output tokens in the decode phase. |
/// in_k_cache         | ^            | Refer to `atb_speed::common::AddSelfAttention` in the `operations/fusion/attention/self_attention.h` for more details. |||
/// in_v_cache         | ^            | ^ |||
/// in_attention_mask  | ^            | ^ |||
/// in_token_offset    | ^            | ^ |||
/// in_layer_id        | ^            | ^ |||
/// in_block_tables    | ^            | ^ |||
/// in_slots_in_pa_or_logn_in_fa | ^  | ^ |||
/// in_weight_dense    | ^            | Weights for the dense linear. Refer to `atb_speed::common::LinearParallel` in the `operations/fusion/linear/linear_parallel.h` for more details. |||
/// in_scale_dense     | ^            | ^ |||
/// in_offset_dense    | ^            | ^ |||
/// in_descale_dense   | ^            | ^ |||
/// in_bias_dense      | ^            | ^ |||
/// in_compress_idx_dense | ^         | ^ |||
/// in_slopes          | `param.selfAttentionParam.maskType` is in one of `atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS`, `atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT` and `atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN` | Refer to `atb_speed::common::AddSelfAttention` in the `operations/fusion/attention/self_attention.h` for more details. |||
/// in_batch_wins      | `param.pageAttentionParam.compressType` equals to `atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD` or `atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE` | ^ |||
/// in_ra_seq_len      | ^            | ^ |||
/// in_pffset_index    | `param.pageAttentionParam.compressType == atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE` | ^ |||
/// in_ra_offset       | ^            | ^ |||
/// in_reshape_seq_len | ^            | ^ |||
/// in_q_len           | `param.pageAttentionParam.calcType == atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC` | ^ |||
/// in_k_quant_scale   | `param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION` | float16/bfloat16 | [head_num * head_dim] | |
/// in_k_dequant_scale | ^            | Refer to `atb_speed::common::AddSelfAttention` in the `operations/fusion/attention/self_attention.h` for more details. |||
/// in_v_quant_scale   | ^            | float16/bfloat16 | [head_num * head_dim] | |
/// in_v_dequant_scale | ^            | Refer to `atb_speed::common::AddSelfAttention` in the `operations/fusion/attention/self_attention.h` for more details. |||
/// in_k_quant_offset  | `param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION` and `param.pageAttentionParam.hasQuantOffset` is true | int8 | [head_num * head_dim] | |
/// in_k_dequant_offset| ^            | Refer to `atb_speed::common::AddSelfAttention` in the `operations/fusion/attention/self_attention.h` for more details. |||
/// in_v_quant_offset  | ^            | int8 | [head_num * head_dim] | |
/// in_v_dequant_offset| ^            | Refer to `atb_speed::common::AddSelfAttention` in the `operations/fusion/attention/self_attention.h` for more details. |||
/// in_q_quant_scale   | `param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE` | float16 | [head_num,head_dim] | |
/// in_k_quant_scale   | ^            | float16 | [kv_head_num,head_dim] | |
/// in_v_quant_scale   | ^            | float16 | [kv_head_num,head_dim] | |
/// in_qk_descale      | ^            | Refer to `atb_speed::common::AddSelfAttention` in the `operations/fusion/attention/self_attention.h` for more details. |||
/// q_offset           | ^            | int8 | [head_num,head_dim] | |
/// kv_offset          | ^            | int8 | [kv_head_num,head_dim] | |
/// fa3_v_quant_scale  | ^            | Refer to `atb_speed::common::AddSelfAttention` in the `operations/fusion/attention/self_attention.h` for more details. |||
/// fa3_offset         | ^            | int32 | [head_num] | |
/// in_im_mask         | `param.supportLora` is true and `param.useImMask` is true | Refer to atb_speed::common::QKVLinearSplit in the `operations/fusion/attention/qkv_linear_split.h` for more details. |||
/// in_seq_len_cum_sum | `param.supportLora` is true | ^ |||
/// in_lora_a_0        | ^            | ^ |||
/// in_lora_b_0        | ^            | ^ |||
/// in_lora_a_1        | ^            | ^ |||
/// in_lora_b_1        | ^            | ^ |||
/// in_lora_a_2        | ^            | ^ |||
/// in_lora_b_2        | ^            | ^ |||
/// in_dense_lora_a    | ^            | ^ |||
/// in_dense_lora_b    | ^            | ^ |||
/// in_reduce_quant_scale | `param.selfOutLinearTensorParallelInfo.quantType != atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED` | Refer to `atb_speed::common::LinearParallel` in the `operations/fusion/linear/linear_parallel.h` for more details. |||
/// in_reduce_quant_offset| ^         | ^ |||
/// in_gather_quant_scale | ^         | ^ |||
/// in_gather_quant_offset| ^         | ^ |||
/// in_log_n_scale     | `param.pageAttentionParam.scaleType == atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_LOGN` | Refer to `atb_speed::common::AddSelfAttention` in the `operations/fusion/attention/self_attention.h` for more details. |||
/// in_q_norm_weight   | `param.useQKNorm` is true | Refer to atb_speed::common::QKVLinearSplit in the `operations/fusion/attention/qkv_linear_split.h` for more details. |||
/// in_k_norm_weight   | ^            | ^ |||
/// in_residual_add    | `param.enableAddNorm` is true | ^ |||
///
/// Operations's outputs:
/// Name       | Dtype                | Shape                |
/// -----------|----------------------|----------------------|
/// out        | The same as in_input | The same as in_input |
/// out_add    | The same as in_input | The same as in_input |
///
/// Example:
/// \code
/// atb::Node fusionAttentionNode;
/// atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
/// // Modify fusionAttentionParam's attribute if needed.
/// Attention(fusionAttentionParam, &fusionAttentionNode.operation);
/// fusionAttentionNode.inTensorIds = {...};  // Passing inputs for the operation in order
/// fusionAttentionNode.outTensorIds = {...};  // Tensor index for out
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(fusionAttentionNode);
/// \endcode
template <typename NormParamType>
atb::Status Attention(const FusionAttentionParam<NormParamType> &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif