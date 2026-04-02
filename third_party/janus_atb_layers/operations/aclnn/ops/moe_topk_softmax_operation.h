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
#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_TOPK_SOFTMAX_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_TOPK_SOFTMAX_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

struct MoeTopkSoftmaxParam {
    /// The number of experts selected for each token
    int64_t topkNum = 2;
};

/// This class defines an operator that first applies softmax to each row of the input, and then
/// selects the top k greatest value.
///
/// This class makes uses of `aclnnMoeGatingTopKSoftmaxGetWorkspaceSize` and `aclnnMoeGatingTopKSoftmax`
/// from the AscendCL API.
///
/// Inputs to the operator:
/// Name            | Dtype                 | Shape |
/// ----------------|-----------------------|-------|
/// input           | float16 or bfloat16   | [m,e] |
///
/// Outputs of the operator:
/// Name            | Dtype                 | Shape |
/// ----------------|-----------------------|-------|
/// output          | float16 or bfloat16   | [m,k] |
/// expertIdx       | int32                 | [m,k] |
/// rowIdx          | int32                 | [m,k] |
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     OUT,
///     OUT_EXPERTIDX,
///     OUT_ROWIDX
/// };
///
/// atb::Node &topKNode = opGraph.nodes.at(nodeId++);
/// atb_speed::common::MoeTopkSoftmaxParam moeTopkSoftmaxParam;
/// moeTopkSoftmaxParam.topkNum = int64_t(param.num.at(0));
/// topKNode.operation = new atb_speed::common::MoeTopkSoftmaxOperation("MoeTopkSoftmaxOperation", moeTopkSoftmaxParam);
/// topKNode.inTensorIds = {INPUT};
/// topKNode.outTensorIds = {OUT,
///                          OUT_EXPERTIDX,
///                          OUT_ROWIDX};
///
/// \endcode

class MoeTopkSoftmaxOperation : public AclNNOperation {
public:
    explicit MoeTopkSoftmaxOperation(const std::string &name, MoeTopkSoftmaxParam param);
    ~MoeTopkSoftmaxOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    MoeTopkSoftmaxParam param_;
};

}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_MOE_TOPK_SOFTMAX_OPERATION_H