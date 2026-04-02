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

#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_TOKEN_UNPERMUTE_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_TOKEN_UNPERMUTE_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"


/// This class defines an operator that is used to gather and reduce hidden states based on sortedIndices.
///
/// This class makes uses of `aclnnMoeTokenUnpermuteGetWorkspaceSize` and `aclnnMoeTokenUnpermute`
/// from the AscendCL API.
///
/// Inputs to the operator:
/// Name           | Dtype               | Shape   |
/// ---------------|---------------------|---------|
/// permutedTokens | float16 or bfloat16 | [m*k,h] |
/// sortedIndices  | int32               | [m*k]   |
/// expertsWeights | float16 or bfloat16 | [m,k]   |
///
/// Outputs of the operator:
/// Name                        | Dtype | Shape   |
/// ----------------------------|-------|---------|
/// out                         | int32 | [m*k,h] |
/// Note:  k is the number of experts selected for each token
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_IDX,
///     IN_EXPERT_WEIGHT,
///     OUT_MOE_MLP_RESULT,
/// };
///
/// atb::Node &unpermuteNode = opGraph.nodes.at(nodeId++);
/// unpermuteNode.operation = new atb_speed::common::MoeTokenUnpermuteOperation("MoeTokenUnpermuteNode");
/// unpermuteNode.inTensorIds = {IN_INPUT,
///                              IN_IDX,
///                              IN_EXPERT_WEIGHT};
/// unpermuteNode.outTensorIds = {OUT_MOE_MLP_RESULT};
/// \endcode

namespace atb_speed::common {
    class MoeTokenUnpermuteOperation : public AclNNOperation {
    public:
        explicit MoeTokenUnpermuteOperation(const std::string &name);
        ~MoeTokenUnpermuteOperation() override;
        atb::Status InferShape(
            const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs
        ) const override;
        [[nodiscard]] uint32_t GetInputNum() const override;
        [[nodiscard]] uint32_t GetOutputNum() const override;

    protected:
        int SetAclNNWorkspaceExecutor() override;
        int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    };
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_MOE_TOKEN_UNPERMUTE_OPERATION_H