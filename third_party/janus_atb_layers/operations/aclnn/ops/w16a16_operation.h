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
#ifndef ATB_SPEED_PLUGIN_ACLNN_W16A16_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_W16A16_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "aclnnop/aclnn_addmm.h"

namespace atb_speed {
namespace common {

/// A struct defines `W16A16Operation`'s parameter.
struct AclNNMatmulParam {
    /// A flag indicating whether the second matrix in the matmul operation is transposed.
    bool transposeB = false;
    /// A flag indicating whether the matmul operation includes a bias tensor.
    bool hasBias = false;
};

/// This class defines a matrix operation combines the matmul and add bias operation.
///
/// This class makes use of `aclnnAddmmGetWorkspaceSize` and `aclnnAddmm` from the AscendCL API.
///
/// Operation's Inputs:
/// Name            | Dtype                       | Shape | Description |
/// ----------------|-----------------------------|-------|-------------|
/// input           | FLOAT, FLOAT16, BFLOAT16    | [m,k] | |
/// weight          | FLOAT, FLOAT16, BFLOAT16    | [n,k] if `transposeB` is true; otherwise, [k,n] | |
/// bias            | FLOAT, FLOAT16, BFLOAT16    | [m,n] or can be broadcasted to [m,n] | Optional. Required if `hasBias` is true. |
///
/// Operations's Outputs:
/// Name   | Dtype                              | Shape |
/// -------|------------------------------------|-------|
/// out    | the same dtype as the input tensor | [m,n] |
///
/// Example:
/// \code
/// enum InTensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_WEIGHT,
///     IN_BIAS,
///     OUT,
/// };
///
/// atb::Node linearNode;
/// AclNNMatmulParam aclNNMatmulParam;
/// aclNNMatmulParam.hasBias = false;
/// aclNNMatmulParam.transposeB = true;
/// linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT};
/// linearNode.outTensorIds = {OUT};
/// linearNode.operation = new atb_speed::common::W16A16Operation("W16A16LinearNode", aclNNMatmulParam);
///
/// atb::Node linearWithBiasNode;
/// AclNNMatmulParam aclNNMatmulParam;
/// aclNNMatmulParam.hasBias = true;
/// aclNNMatmulParam.transposeB = true;
/// linearWithBiasNode.inTensorIds = {
///     IN_INPUT, IN_WEIGHT, IN_BIAS};
/// linearWithBiasNode.outTensorIds = {OUT};
/// linearWithBiasNode.operation = new atb_speed::common::W16A16Operation(
///     "W16A16LinearWithBiasNode", aclNNMatmulParam);
///
/// // Add the operation node to the graph as required
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(linearNode);
/// opGraph.nodes.push_back(linearWithBiasNode);
/// \endcode

class W16A16Operation : public AclNNOperation {
public:
    explicit W16A16Operation(const std::string &name, AclNNMatmulParam param);
    ~W16A16Operation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    int CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    int CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;

    atb::Dims GetWeightStorageShape(const atb::TensorDesc atbTensorDesc) const;

    AclNNMatmulParam param_;
};
}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PUBLIC_ACLNN_W8A8_OPERATION_H