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
#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_FUSED_REDUCESUM_DIV_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_FUSED_REDUCESUM_DIV_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

//TODO(panxuanyu): update docs
/// This class defines an operator that is used to gather and rearrange hidden states based
/// on the given list of selected experts of each token.
///
/// This class makes uses of `aclnnMoeFusedReducesumDivGetWorkspaceSize` and `aclnnMoeFusedReducesumDiv`
/// from the AscendCL API.
///
/// Inputs to the operator:
/// Name         | Dtype               | Shape |
/// -------------|---------------------|-------|
/// input        | float16 or bfloat16 | [m,h] |
/// expertIdx    | int32               | [m,k] |
///
/// Outputs of the operator:
/// Name                         | Dtype | Shape   |
/// -----------------------------|-------|---------|
/// expandedXOut                 | int32 | [m*k,h] |
/// expandedRowIdxOut            | int32 | [m*k]   |
/// expertTokensCountOrCumsumOut | int32 | [e]     |
/// Note: e is the total number of experts utilized by the model
/// k is the number of experts selected for each token
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_EXPERTIDX,
///     OUT_SORTED_HIDDENSTATES,
///     OUT_ROWIDX,
///     OUT_GROUP_LIST
/// };
///
/// atb::Node &initRoutingNode = opGraph.nodes.at(nodeId++);
/// atb_speed::common::MoeFusedReducesumDivParam initRoutingParam;
/// initRoutingParam.topkNum = param.topk;
/// initRoutingParam.expertNum = param.numOfExperts;
/// initRoutingNode.operation = new atb_speed::common::MoeFusedReducesumDivOperation("MoeFusedReducesumDivOperation",
///                                                                            initRoutingParam);
/// initRoutingNode.inTensorIds = {IN_PUT, IN_EXPERTIDX};
/// initRoutingNode.outTensorIds = {OUT_SORTED_HIDDENSTATES,
///                                 OUT_ROWIDX,
///                                 OUT_GROUP_LIST};
/// \endcode

class MoeFusedReducesumDivOperation : public AclNNOperation {
public:
    explicit MoeFusedReducesumDivOperation(const std::string &name);
    ~MoeFusedReducesumDivOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
};

}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_MOE_TOPK_SOFTMAX_OPERATION_H
