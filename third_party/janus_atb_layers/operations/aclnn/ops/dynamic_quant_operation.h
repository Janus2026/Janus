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
#ifndef ATB_SPEED_PLUGIN_ACLNN_DYNAMIC_QUANT_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_DYNAMIC_QUANT_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"

/// This class defines an dynamic quant operator.
///
/// This class makes uses of `aclnnDynamicQuantGetWorkspaceSize` and `aclnnDynamicQuant`
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
/// out            | int8                | [m,h]   |
/// tokenScales    | float16 or bfloat16 | [m]     |
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     OUT,
///     OUT_SCALE,
/// };
///
/// atb::Node &dynamicQuantNode = opGraph.nodes.at(nodeId++);
/// dynamicQuantNode.operation = new atb_speed::common::DynamicQuantOperation("DynamicQuantOperation");
/// dynamicQuantNode.inTensorIds = {IN_INPUT};
/// dynamicQuantNode.outTensorIds = {OUT, OUT_SCALE};
/// \endcode

namespace atb_speed::common {

class DynamicQuantOperation : public AclNNOperation {
public:
    explicit DynamicQuantOperation(const std::string &name);
    ~DynamicQuantOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                            atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
};
}

#endif  // ATB_SPEED_PLUGIN_ACLNN_DYNAMIC_QUANT_OPERATION_H