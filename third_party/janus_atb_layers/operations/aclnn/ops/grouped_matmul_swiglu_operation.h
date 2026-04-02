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
#ifndef ATB_SPEED_PLUGIN_ACLNN_GROUPED_MATMUL_SWIGLU_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_GROUPED_MATMUL_SWIGLU_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {
enum GmmQuantSwigluType : int {
    NONE1 = 0,
    W8A8_CHANNEL1,
    W8A16_CHANNEL1,
    W8A8_TOKEN1
};

struct AclNNGroupedSwigluMatmulParam {
    bool transposeB = false;  /// A flag indicating wheter the second input matrix needs to be transposed
    int quantType = 0;  /// The quantization type of the operation
    bool hasBias = false;  /// A flag indicating whether the matmul operation includes a bias tensor
    aclDataType outDataType = ACL_FLOAT16;  /// The data type of the outpuot of the oepration
};

/// This calss defines an operator that consists of a group of matrix multiplications.
/// Meanwhile, this operator supports different quantization types.
///
/// This class makes uses of `aclnnGroupedMatmulV4GetWorkspaceSize` and `aclnnGroupedMatmulV4`
/// from the AscendCL API.
///
/// Inputs to the operator:
/// Name                    | Dtype | Shape |
/// ------------------------|-------|-------|
/// input                   | *     | [m,k] |
/// weight                  | *     | [e,n,k] if `transposeB` is true; otherwise, [e,k,n] |
/// PerChannelscale           | *     | [e,k] |
/// PerTokenscale           | *     | [m] |
/// groupList               | int64 | [e]   |
/// * Note: the data type of inputs are speccfic to the quantization type/technique chosen for the model
///
/// Outputs of the operator:
/// Name   | Dtype               | Shape |
/// -------|---------------------|-------|
/// quant_output | int8 | [m,n/2] |
/// quant_scale_output | float | [m] |
///
/// Example:
/// \code
/// enum InTensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_WEIGHT,
///     IN_SCALE_EXPERT,
///     IN_DYNAMIC_SCALE,
///     IN_GROUP_LIST,
/// };
///
/// enum OutTensorIdx : uint32_t {
///     QUANT_OUT = 0,
///     QUANT_SCALE
/// };
///
/// atb::Node &gmmNode = opGraph.nodes.at(nodeId++);
/// atb_speed::common::AclNNGroupedMatmulParam gmmParam;
/// gmmParam.quantType = gmmQuantType;
/// gmmParam.outDataType = param.outDataType;
/// gmmParam.transposeB = param.transposeB;
/// gmmNode.operation = new atb_speed::common::GroupedMatmulSwigluOperation("gmmNode", gmmParam);
/// gmmNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_SCALE_EXPERT, IN_DYNAMIC_SCALE, IN_GROUP_LIST};
/// gmmNode.outTensorIds = {QUANT_OUT,QUANT_SCALE};
/// \endcode

class GroupedMatmulSwigluOperation : public AclNNOperation {
public:
   explicit GroupedMatmulSwigluOperation(const std::string &name, AclNNGroupedSwigluMatmulParam param);
    ~GroupedMatmulSwigluOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
    AclNNGroupedSwigluMatmulParam param_;
    static constexpr uint32_t INPUT_NUM = 5U;
    static constexpr uint32_t OUTPUT_NUM = 2U;
};
}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_GROUPED_MATMUL_OPERATION_H
