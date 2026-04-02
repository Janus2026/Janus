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
#ifndef ATB_SPEED_MODELS_COMMON_LINEAR_H
#define ATB_SPEED_MODELS_COMMON_LINEAR_H

#include "atb/atb_infer.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/utils.h"
#include "operations/fusion/parallel_info.h"

namespace atb_speed {
namespace common {


/// Parameters for the fusion linear module
struct FusionLinearParam {
    /// Specifies how the linear module is quantized.
    /// Refer to the `LinearQuantType` definition in the operations/utils.h.
    LinearQuantType quantType = NO_QUANT;
    /// When `isBF16` is true, bfloat16 precision is used; otherwise, float16 precision is used.
    bool isBF16 = false;
    /// Specifies whether linear module has bias.
    bool hasBias = false;
    /// A flag indicating whether lora is enabled.
    bool supportLora = false;
    /// A flag indicating whether a mask is used before applying lora adapter.
    bool useImMask = false;
    /// A flag indicating whether the group matmul operation is enabled;
    /// it should be activated when batch inputs include multiple LoRA adapters
    bool loraEnableGMM = false;
    /// Defines whether the second matrix in the matmul operation is transposed.
    int transposeType = TRANSPOSE;
    /// The group size used for dequantizing the weight tensor in the per-group quantization approach
    int quantGroupSize = 0;
    /// A flag indicating whether to use the atb matmul backend
    int matmulBackend = atb_speed::common::OpBackend::ATB;
    /// A flag indicating whether to use EinMatmul;
    bool enEin = false;
    /// A flag indicating whether throw dequant.
    bool isThrowDequant = false;
    /// A flag indicating the prefill and decode phases
    bool isPrefill = false;
    /// A flag indicating whether the linear operation throws out dequant operation.
    bool enableDequantBias = false;
    /// A flag indicating whether the model use cube and vector parallel
    bool enableCVOverlap = false;

    bool enableSwiGLUQuantForSharedExperts = false;
    /// A flag indicating whether to use swigluQuant
    bool enableSwigluQuant = false;
    /// A flag indicating whether is down linear
    bool isDownLinear = false;
    /// for flashcomm 1.0
    TensorParallelInfo flashCommParallelInfo;
    bool enableFlashComm = false;
};

/// Check whether is w8a8 per tensor and matmulBackend is ACLNN
///
/// \param FusionLinearParam The linear params to check
/// \return True if matmulBackend is ACLNN under w8a8 per tensor
bool IsAclnnPerTensor(const FusionLinearParam &param);

/// Check whether is w8a8 dynamic scene, plus the conditions include aclnnPerTensor
/// Return A flag if use aclnn operator QuantBatchMatmul
///
/// \param FusionLinearParam The linear params to check
/// \return True if quantType is DYNAMIC or ACLNN with W8A8
bool UseQuantBatchMatmul(const FusionLinearParam &param);

/// This function is the main entrance for all types of linear modules.
/// It will call different operations based on the `quantType`.
/// Note that linear module with `quantType` equals to `LINEAR_W8A8_DYNAMIC_DEQUANT` is not implemented yet.
///
/// \param param Parameters for the fusion linear module
/// \param operation the address of a pointer to a default operation
/// \return A flag that indicates whether operation has been successfully created.
///
/// Operation's inputs when `quantType` is `NO_QUANT`:
/// Name            | Dtype                    | Shape |
/// ----------------|--------------------------|-------|
/// in_input        | float16/bfloat16         | [m,k] |
/// in_weight       | float16/bfloat16         | [n,k] if `transposeB` is true; otherwise, [k,n] |
/// in_scale        | float16                  | [1]   |
/// in_offset       | float16                  | [1]   |
/// in_descale      | float16                  | [1]   |
/// in_bias         | float16                  | [1]   |
/// in_compress_idx | float16                  | [1]   |
///
/// Operation's inputs when `quantType` is `LINEAR_W8A8_DEQUANT`:
/// Name            | Dtype                    | Shape |
/// ----------------|--------------------------|-------|
/// in_input        | int8                     | [m,k] |
/// in_weight       | int8                     | [n,k] if `transposeB` is true; otherwise, [k,n] |
/// in_scale        | float16                  | [1]   |
/// in_offset       | float16                  | [1]   |
/// in_descale      | int64 if the output tensor's dtype is float16; float32 if the output tensor's dtype is bfloat16 | [n] |
/// in_bias         | int32                    | [n]   |
/// in_compress_idx | float16                  | [1]   |
///
/// Operation's inputs when `quantType` is `LINEAR_W8A8_QUANT`:
/// Name            | Dtype                    | Shape |
/// ----------------|--------------------------|-------|
/// in_input        | float16/bfloat16         | [m,k] |
/// in_weight       | int8                     | [n,k] if `transposeB` is true; otherwise, [k,n] |
/// in_scale        | the same dtype as in_input | [1]   |
/// in_offset       | the same dtype as in_input | [1]   |
/// in_descale      | int64 if the output tensor's dtype is float16; float32 if the output tensor's dtype is bfloat16 | [n] |
/// in_bias         | int32                    | [n]   |
/// in_compress_idx | float16                  | [1]   |
///
/// Operation's inputs when `quantType` is `W4A16`:
/// Name            | Dtype                    | Shape | Description |
/// ----------------|--------------------------|-------|---------|
/// in_input        | int8                     | [m,k] |         |
/// in_weight       | int8                     | [n,k/2] if `transposeB` is true; otherwise, [k,n/2] |
/// in_scale        | the same dtype as the output tensor | [n,1]/[n,ceil(k, group_size)] if `transposeB` is true; otherwise, [1,n]/[ceil(k, group_size),n] | |
/// in_offset       | the same dtype as the output tensor | [n,1]/[n,ceil(k, group_size)] if `transposeB` is true; otherwise, [1,n]/[ceil(k, group_size),n] | |
/// in_descale      | float16                  | [1]   |         |
/// in_bias         | int32 if the output tensor's dtype is float16; bfloat16 if the output tensor's dtype is bfloat16 | [n]   | Used when `hasBias` is true. |
/// in_compress_idx | float16                  | [1]   |         |
///
/// Operation's inputs when `quantType` is `W8A16`:
/// Name            | Dtype                    | Shape | Description |
/// ----------------|--------------------------|-------|---------|
/// in_input        | int8                     | [m,k] |         |
/// in_weight       | int8                     | [n,k] if `transposeB` is true; otherwise, [k,n] | |
/// in_scale        | the same dtype as the output tensor | [n,1]/[n,ceil(k, group_size)] if `transposeB` is true; otherwise, [1,n]/[ceil(k, group_size),n] | |
/// in_offset       | the same dtype as the output tensor | [n,1]/[n,ceil(k, group_size)] if `transposeB` is true; otherwise, [1,n]/[ceil(k, group_size),n] | |
/// in_descale      | float16                  | [1]   |         |
/// in_bias         | int32 if the output tensor's dtype is float16; bfloat16 if the output tensor's dtype is bfloat16 | [n]   | Used when `hasBias` is true. |
/// in_compress_idx | float16                  | [1]   |         |
///
/// Operation's inputs when `quantType` is `LINEAR_W8A8_SC_DEQUANT`:
/// Name            | Dtype                    | Shape |
/// ----------------|--------------------------|-------|
/// in_input        | int8                     | [m,k] |
/// in_weight       | int8                     | One dimensional tensor with variable shape |
/// in_scale        | float16                  | [1]   |
/// in_offset       | float16                  | [1]   |
/// in_descale      | int64                    | [n]   |
/// in_bias         | int32                    | [n]   |
/// in_compress_idx | int8                     | One dimensional tensor with variable shape |
///
/// Operation's inputs when `quantType` is `LINEAR_W8A8_SC_QUANT`:
/// Name            | Dtype                    | Shape |
/// ----------------|--------------------------|-------|
/// in_input        | float16                  | [m,k] |
/// in_weight       | int8                     | One dimensional tensor with variable shape |
/// in_scale        | float16                  | [1]   |
/// in_offset       | float16                  | [1]   |
/// in_descale      | int64                    | [n]   |
/// in_bias         | int32                    | [n]   |
/// in_compress_idx | int8                     | One dimensional tensor with variable shape |
///
/// Operation's inputs when `quantType` is `LINEAR_W8A8_DYNAMIC_QUANT`:
/// Name            | Dtype                    | Shape |
/// ----------------|--------------------------|-------|
/// in_input        | float16                  | [m,k] |
/// in_weight       | int8                     | [n,k] if `transposeB` is true; otherwise, [k,n] |
/// in_scale        | float16                  | [n,1] |
/// in_offset       | float16                  | [n,1] |
/// in_descale      | float16                  | [1]   |
/// in_bias         | float16                  | [1]   |
/// in_compress_idx | float16                  | [1]   |
///
/// Operation's optional inputs:
/// Name            | Dtype           | Shape       | Condition     |
/// ----------------|-----------------|-------------|---------------|
/// in_im_mask      | float16         | [m,1]       | Required and used when `supportLora` and `useImMask` is true |
/// in_group_list   | int64           | [batchSize] | Required when `supportLora` is true and only used when `loraEnableGMM` is true |
/// in_lora_a       | float16/bfloat16 | [r,k] if `transposeB` is true; otherwise, [k,r] | Required and used when `supportLora` is true |
/// in_lora_b       | float16/bfloat16 | [n,r] if `transposeB` is true; otherwise, [r,n] | Required and used when `supportLora` is true |
///
/// Operation's Outputs:
/// Name   | Dtype               | Shape |
/// -------|---------------------|-------|
/// out    | float16/bfloat16    | [m,n] |
/// Note that operations with `quantType` equals to `LINEAR_W8A8_DYNAMIC_QUANT`, `LINEAR_W8A8_SC_DEQUANT`
/// and `LINEAR_W8A8_SC_QUANT` do not support bfloat16.
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_WEIGHT,
///     IN_PLACEHOLDER,
///     OUT,
/// };
///
/// atb::Node linearNode;
/// atb_speed::common::FusionLinearParam linearParam;
/// // Modify linearParam's attribute if needed.
/// FusionLinear(linearParam, &linearNode.operation);
/// linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_PLACEHOLDER, IN_PLACEHOLDER, IN_PLACEHOLDER, IN_PLACEHOLDER, IN_PLACEHOLDER};
/// linearNode.outTensorIds = {OUT};
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(linearNode);
/// \endcode
atb::Status FusionLinear(const FusionLinearParam &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif