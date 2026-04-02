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
#ifndef ATB_SPEED_MODELS_COMMON_MLP_OPERATION_H
#define ATB_SPEED_MODELS_COMMON_MLP_OPERATION_H
#include <map>
#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {

/// The categories of the mlp module's input tensors
/// Input tensors will be arragned according to the order of their categories
enum MlpInTensorCategory : unsigned int {
    /// Default tensors
    MLP_DEFAULT = 0,
    /// Tensors required by addRmsNormQuant, addRmsNormDynamicQuant
    MLP_ADD_RMS_NORM_QUANT,
    /// Tensors required by the add norm fusion operation
    MLP_ADD_NORM,
    /// The mask tensor before applying lora adapters
    MLP_LORA_MASK,
    /// Tensors needed for LoRA
    MLP_LORA,
    /// Tensors required by the quantization of the all reduce operation
    MLP_REDUCE_QUANT,
    /// Tensors needed for Flashcomm1.0
    MLP_FC,
    /// A flag signifying the end of all categories
    MLP_END
};

/// The pack type of the gate and up linear
enum MlpPackType : unsigned int {
    /// The gate and up linear is packed
    GATE_UP_WEIGHT_PACK = 0,
    /// The gate and up linear is not packed
    GATE_UP_WEIGHT_NO_PACK = 1,
    /// No gate linear
    UP_WEIGHT_ONLY = 2,
};

/// The index of the gate linear within the layer
const uint64_t GATE_LINEAR_INDEX = 4;
/// The index of the up linear within the layer
const uint64_t UP_LINEAR_INDEX = 5;
/// The index of the down linear within the layer
const uint64_t DOWN_LINEAR_INDEX = 6;

/// Parameters for the mlp module
/// \tparam NormParamType Types of the normalization parameters. Avaliable options are `atb::infer::RmsNormParam`
/// and `atb::infer::LayerNormParam`.
template <typename NormParamType>
struct MlpParam {
    /// When `isBF16` is true, bfloat16 precision is used; otherwise, float16 precision is used.
    bool isBF16 = false;
    /// A flag indicating the prefill and decode phases
    bool isPrefill = false;
    bool isEdgeHardware = false;
    /// A flag indicating whether gate and up linear has bias
    bool gateUpHasBias = false;
    /// A flag indicating whether down linear has bias
    bool downHasBias = false;
    /// A flag that indicates whether low-latency computation over communication is enabled
    bool supportLcoc = false;
    bool enableMC2 = false;
    /// A flag indicating whether normalization is skipped
    bool skipNorm = false;
    /// A flag indicating whether normalization has bias
    bool normHasBias = false;
    /// A flag indicating whether to use the AddNorm fusion operation
    bool enableAddNorm = false;
    /// A flag indicating whether to use NormQuant fusion operation
    bool enableNormQuantOp = true;
    /// A flag indicating whether lora is enabled.
    bool supportLora = false;
    /// A flag indicating whether a mask is used before applying lora adapter.
    bool useImMask = false;
    /// it should be activated when batch inputs include multiple LoRA adapters
    bool loraEnableGMM = false;
    /// A flag indicating whether to use swigluQuant
    bool enableSwigluQuant = false;
    /// A flag indicating whether to use flashcomm 1.0
    bool enableFlashComm = false;
    /// The pack type of the gate and up linear. Refer to `MlpPackType` in the `operations/mlp/mlp.h`.
    MlpPackType mlpPackType = GATE_UP_WEIGHT_PACK;
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
    /// Indicates the pack type and the quantization type of the gate up linear.
    int packQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    /// The group size used for dequantizing the weight tensor in the per-group quantization approach
    int quantGroupSize = 0;
    /// Normalization parameters for float operation
    NormParamType normParamType;
    /// Normlization parameters for quantization operation
    NormParamType normQuantParamType;
    /// Parameters for the activation operation
    atb::infer::ActivationParam activationParam;
    /// The quantization type of the down linear. Refer to `PackQuantType` in the `operations/utils.h`.
    int downQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    /// Details about tensor parallelism
    atb_speed::common::TensorParallelInfo downLinearTensorParallelInfo;
    /// A flag indicating whether to use the atb matmul backend
    int matmulBackend = atb_speed::common::OpBackend::ATB;
    /// Specifies whether the post attention norm enables antioutlier
    bool isAntiOutlier = false;
    // Use Aclnn RmsNorm instead of ATB RmsNorm.
    bool enableAclnnRmsNorm = false;
};

/// Get the `MlpPackType` based on the quantizaton type of the gate-up linear and the structure of the model.
/// \param packQuantType Parameters to determin whether the gate-up linear is packed.
///     Refer to `PackQuantType` in the `operations/utils.h`.
/// \param upWeightOnly A flag indicating if the structure of the layer only has up linear.
/// \param linearDescs weight description of linear module
/// \return Refer to `MlpPackType` in the `operations/fusion/mlp.h`.
MlpPackType GetMlpPackType(
    const int &packQuantType = PackQuantType::PACK_QUANT_UNDEFINED,
    bool upWeightOnly = false,
    const std::vector<int> &linearDescs = {});

/// The mlp module.
/// It consists of the following operations: NormLinear operations for the gate-up linear,
/// Split operation if the gate-up linear is packed, Activation operation, Elementwise mul operation
/// and LinearParallel operation for the down linear.
///
/// \tparam NormParamType Types of the normalization parameters. Avaliable options are `atb::infer::RmsNormParam`
/// and `atb::infer::LayerNormParam`.
/// \param param Parameters for the mlp module
/// \param operation The address of a pointer to a default operation
/// \return A flag that indicates whether operation has been successfully created.
///
/// Operation's inputs:
/// Name                   | Requirements | Dtype            | Shape | Description |
/// -----------------------|--------------|------------------|-------|----------|
/// in_input               | Required     |float16/bfloat16 | paged attention: [len(all_seq),hidden_size] | Hidden states |
/// ^                      | ^            | ^               | flash attention: [bsz,seq_len,hidden_size]  | ^             |
/// in_norm_weight         | ^            | Refer to `NormLinear` in the `operations/fusion/norm/norm_linear.h` for more details. |||
/// in_norm_bias           | ^            | ^ |||
/// in_norm_new_weight     | ^            | ^ |||
/// in_norm_new_bias       | ^            | ^ |||
/// in_weight_0            | ^            | If gate-up linear are packed, these are the concatenated gate-up weights. <br> If the gate-up linear is unpacked, these are weights for the gate linear. <br> Weights for the up linear will be passed in if the layer only has up weights <br> Refer to `NormLinear` in the `operations/fusion/norm/norm_linear.h` for more details. |||
/// in_scale_0             | ^            | ^ |||
/// in_offset_0            | ^            | ^ |||
/// in_descale_0           | ^            | ^ |||
/// in_bias_0              | ^            | ^ |||
/// in_compress_idx_0      | ^            | ^ |||
/// in_weight_1            | ^            | If gate-up linear are not packed, these are weights for the up linear operation; otherwise, placeholders should be provided. |||
/// in_scale_1             | ^            | ^ |||
/// in_offset_1            | ^            | ^ |||
/// in_descale_1           | ^            | ^ |||
/// in_bias_1              | ^            | ^ |||
/// in_compress_idx_1      | ^            | ^ |||
/// in_weight_down         | ^            | Weights for the dense linear operation. |||
/// in_scale_down          | ^            | ^ |||
/// in_offset_down         | ^            | ^ |||
/// in_descale_down        | ^            | ^ |||
/// in_bias_down           | ^            | ^ |||
/// in_compress_idx_down   | ^            | ^ |||
/// in_residual_add        | `param.enableAddNorm` is true | The same as in_input | The same as in_input | |
/// in_im_mask             | when `param.supportLora` and `param.useImMask` are true | Refer to `FusionLinear` in the `operations/fusion/linear/linear.h`. |||
/// in_seq_len_cum_sum     | `param.supportLora` is true | Refer to `FusionLinear` in the `operations/fusion/linear/linear.h`. |||
/// in_lora_a_0            | ^            | ^ |||
/// in_lora_b_0            | ^            | ^ |||
/// in_lora_a_1            | ^            | ^ |||
/// in_lora_b_1            | ^            | ^ |||
/// in_down_lora_a         | ^            | ^ |||
/// in_down_lora_b         | ^            | ^ |||
/// in_reduce_quant_scale  | `param.tensorParallelInfo.quantType == atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL` | Refer to `LinearParallel` in the `operations/fusion/linear/linear_parallel.h`. |||
/// in_reduce_quant_offset | ^            | ^ |||
/// in_gather_quant_scale  | ^            | ^ |||
/// in_gather_quant_offset | ^            | ^ |||
///
/// Operation's Outputs:
/// Name       | Dtype               | Shape | Description |
/// -----------|---------------------|-------|----------|
/// out_linear | The same as in_input| The same as in_input | Output tensor of the mlp module |
/// out_add    | The same as in_input | The same as in_input | The tensor resulting from adding the input and output of the attention module in a residual connection. Exist when `enableAddNorm` is true. |
///
/// Example:
/// \code
/// atb::Node mlpNode;
/// atb_speed::common::MlpParam<atb::infer::RmsNormParam> mlpParam;
/// // Modify mlpParam's attribute if needed.
/// Mlp(mlpParam, &mlpNode.operation);
/// mlpNode.inTensorIds = {...};  // Passing inputs for the operation in order
/// mlpNode.outTensorIds = {...};  // Tensor index for out
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(mlpNode);
/// \endcode
template <typename NormParamType>
atb::Status Mlp(const MlpParam<NormParamType> &param, atb::Operation **operation);

/// The mlp module implemented with the SwiGLU fusion operation.
/// The SwiGLU fusion operation processes the combined output of the gate and up linear operations as input,
/// intergrating the activation and multiplication operations into a single operation.
///
/// \tparam NormParamType Types of the normalization parameters. Avaliable options are `atb::infer::RmsNormParam`
/// and `atb::infer::LayerNormParam`.
/// \param param Parameters for the mlp module
/// \param operation The address of a pointer to a default operation
/// \return A flag that indicates whether operation has been successfully created.
///
/// Inputs and outputs adhere to the same specification as those of the `Mlp`.
template <typename NormParamType>
atb::Status MlpSwiGLU(const MlpParam<NormParamType> &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif