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

#ifndef ATB_SPEED_PLUGIN_ACLNN_REPEAT_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_REPEAT_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed::common {
struct AclNNRepeatParam {
    std::vector<int64_t> repeatsArray;
};

/// This class defines an repeat operator.
///
/// This class makes uses of `aclnnRepeatGetWorkspaceSize` and `aclnnRepeat`
/// from the AscendCL API.
///
/// Inputs to the operator:
/// Name           | Dtype               | Shape   |
/// ---------------|---------------------|---------|
/// input          | float16 or bfloat16 | [m,h]   |
///
/// Outputs of the operator:
/// Name           | Dtype               | Shape   |
/// ---------------|---------------------|---------|
/// out            | float16 or bfloat16 |[m*k,h*n]|
/// Note: k, n are the repetition times.
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     OUT,
/// };
///
/// atb::Node &repeatNode = opGraph.nodes.at(nodeId++);
/// atb_speed::common::AclNNRepeatParam repeatParam;
/// repeatParam.repeatsArray = param.repeatsArray;
/// repeatNode.operation = new atb_speed::common::RepeatOperation("RepeatOperation", repeatParam);
/// repeatNode.inTensorIds = {IN_INPUT};
/// repeatNode.outTensorIds = {OUT};
/// \endcode

class RepeatOperation : public AclNNOperation {
public:
    explicit RepeatOperation(const std::string &name, AclNNRepeatParam param);
    ~RepeatOperation() override;
    atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs
    ) const override;
    [[nodiscard]] uint32_t GetInputNum() const override;
    [[nodiscard]] uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

private:
    AclNNRepeatParam param_;
    std::string opName_;
};
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_REPEAT_OPERATION_H