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
#ifndef ATB_SPEED_PLUGIN_ACLNN_SIGMOID_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_SIGMOID_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"


/// This class defines an sigmoid operator.
///
/// This class makes uses of `aclnnSigmoidGetWorkspaceSize` and `aclnnSigmoid`
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
/// out            | float16 or bfloat16 | [m,h]   |
///
/// Example:
/// \code
/// enum InTensorIdx : uint32_t {
///     IN_INPUT = 0,
/// };
///
/// enum OutTensorIdx : uint32_t {
///     OUT = 0
/// };
///
/// atb::Node &sigmoidNode = opGraph.nodes.at(nodeId++);
/// sigmoidNode.operation = new atb_speed::common::SigmoidOperation("SigmoidOperation");
/// sigmoidNode.inTensorIds = {IN_INPUT};
/// sigmoidNode.outTensorIds = {OUT};
/// \endcode

namespace atb_speed {
namespace common {

class SigmoidOperation : public AclNNOperation {
public:
    explicit SigmoidOperation(const std::string &name);
    ~SigmoidOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
};

} // namespace common
} // namespace atb_speed
#endif // ATB_SPEED_PLUGIN_ACLNN_SIGMOID_OPERATION_H
