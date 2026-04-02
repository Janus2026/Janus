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
#ifndef ATB_SPEED_PLUGIN_ACLNN_W8A8_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_W8A8_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/fusion/utils.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

/// A struct defines `W8A8Operation`'s parameter.
struct AclNNQuantMatmulParam {
    /// A flag indicating whether the matmul operation includes a bias tensor.
    bool hasBias = false;
    /// A flag indicating whether the second matrix in the matmul operation is transposed.
    bool transposeB = true;
    /// A flag indicating whether to use the atb matmul backend
    int matmulBackend = atb_speed::common::OpBackend::ATB;
    /// A flag indicating whether the matmul operation includes a perTokenScaleOptional tensor.
    bool hasPerTokenScale = false;
    /// A flag indicating whether the tensor type is bfloat16.
    bool isBF16 = true;
    /// A flag indicating whether the matmul operation throws out dequantBias operation.
    bool isOutDequantBias = false;
    /// A flag indicating whether the matmul operation includes an offset tensor.
    bool hasOffset = false;
};

/// This class defines a matrix operation that supports
/// dynamic per-token activation quantization and weight per-channel quantization.
///
/// This class makes use of `aclnnQuantMatmulV4GetWorkspaceSize` and `aclnnQuantMatmulV4` from the AscendCL API.
///
/// Operation's Inputs:
/// Name            | Dtype   | Shape |
/// ----------------|---------|-------|
/// input           | int8    | [m,k] |
/// weight          | int8    | [n,k] if `transposeB` is true; otherwise, [k,n] |
/// weight scale    | float32 if the output tensor's dtype is float16; bfloat16 if the output tensor's dtype is bfloat16 | [n] |
/// per token scale | float32 | [m]   |
/// bias            | int32   | [n]   |
///
/// Operations's Outputs:
/// Name   | Dtype               | Shape |
/// -------|---------------------|-------|
/// output | float16 or bfloat16 | [m,n] |
///
/// Example:
/// \code
/// enum InTensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_WEIGHT,
///     IN_WEIGHT_SCALE,
///     IN_PER_TOKEN_SCALE,
///     IN_BIAS,
///     OUT,
/// };
///
/// atb::Node linearNode;
/// AclNNQuantMatmulParam aclnnQuantMatmulParam;
/// aclnnQuantMatmulParam.hasBias = false;
/// aclnnQuantMatmulParam.transposeB = true;
/// linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_WEIGHT_SCALE, IN_PER_TOKEN_SCALE};
/// linearNode.outTensorIds = {OUT};
/// linearNode.operation = new atb_speed::common::W8A8Operation("W8A8LinearNode", aclnnQuantMatmulParam);
///
/// atb::Node linearWithBiasNode;
/// AclNNQuantMatmulParam aclnnQuantMatmulWithBiasParam;
/// aclnnQuantMatmulWithBiasParam.hasBias = true;
/// aclnnQuantMatmulWithBiasParam.transposeB = true;
/// linearWithBiasNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_WEIGHT_SCALE, IN_PER_TOKEN_SCALE, IN_BIAS};
/// linearWithBiasNode.outTensorIds = {OUT};
/// linearWithBiasNode.operation = new atb_speed::common::W8A8Operation(
///     "W8A8LinearWithBiasNode", aclnnQuantMatmulWithBiasParam);
///
/// // Add the operation node to the graph as required
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(linearNode);
/// opGraph.nodes.push_back(linearWithBiasNode);
/// \endcode
class W8A8Operation : public AclNNOperation {
public:
    explicit W8A8Operation(const std::string &name, AclNNQuantMatmulParam param);
    ~W8A8Operation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    int CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    int CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    int CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Dims GetWeightStorageShape(const atb::TensorDesc atbTensorDesc) const;

private:
    AclNNQuantMatmulParam param_;
};
} // namespace common
} // namespace atb_speed
#endif // ATB_SPEED_PUBLIC_ACLNN_W8A8_OPERATION_H