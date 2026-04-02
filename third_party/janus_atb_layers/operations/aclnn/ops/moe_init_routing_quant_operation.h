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
#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_INIT_ROUTING_QUANT_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_INIT_ROUTING_QUANT_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

struct MoeInitRoutingQuantParam {
    int32_t topkNum = 2;  /// The number of experts selected for each token
    int scaledTopk = -1;  /// The non-deepseek models do not have the scaledTopk feature enabled by default
    bool enableInitRoutingCutoff = false;
    int32_t expertNum = 8; /// The total number of experts utilized by the model
    int32_t quantMode = 1; /// The quant mode: 0 is static quant and 1 is dynamic quant
    int expertTokensCoutOrCumsumFlag = 1;
};

/// This calss defines an operator that is used to gather, rearrage and quantize hidden states based
/// on the given list of selected experts of each token.
///
/// This class makes uses of `aclnnMoeInitRoutingQuantV2GetWorkspaceSize` and `aclnnMoeInitRoutingV2Quant`
/// from the AscendCL API.
///
/// Inputs to the operator:
/// Name         | Dtype               | Shape |
/// -------------|---------------------|-------|
/// input        | float16 or bfloat16 | [m,h] |
/// expertIdx    | int32               | [m,k] |
///
/// Outputs of the operator:
/// Name                          | Dtype               | Shape   |
/// ------------------------------|---------------------|---------|
/// expandedXOut                  | int32               | [m*k,h] |
/// expandedRowIdxOut             | int32               | [m*k]   |
/// expertTokensCoutOrCumsumOut   | int32               | [e]     |
/// expertTokensBeforeCapacityOut | int32               | [e]     |
/// dynamicQuantScaleOut          | float16 or bfloat16 | [m*k]   |
/// Note: e is the total number of experts utilized by the model
/// k is the number of experts selected for each token
///
/// Example:
/// \code
/// enum InTensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_EXPERTIDX,
/// };
///
/// enum OutTensorIdx : uint32_t {
///     OUT_SORTED_HIDDENSTATES = 0,
///     OUT_ROWIDX,
///     OUT_GROUP_LIST,
///     OUT_EXPERT_TOKNENS_BEFORE_CAPACITY,
///     OUT_DYNAMIC_QUANT_SCALE
/// };
///
/// atb::Node &initRoutingNode = opGraph.nodes.at(nodeId++);
/// atb_speed::common::MoeInitRoutingQuantParam initRoutingParam;
/// initRoutingParam.topkNum = param.topk;
/// initRoutingParam.expertNum = param.numOfExperts;
/// initRoutingNode.operation = new atb_speed::common::MoeInitRoutingQuantOperation("MoeInitRoutingQuantOperation",
///                                                                            initRoutingParam);
/// initRoutingNode.inTensorIds = {IN_PUT, IN_EXPERTIDX};
/// initRoutingNode.outTensorIds = {OUT_SORTED_HIDDENSTATES,
///                                 OUT_ROWIDX,
///                                 OUT_GROUP_LIST,
///                                 OUT_EXPERT_TOKNENS_BEFORE_CAPACITY,
///                                 OUT_DYNAMIC_QUANT_SCALE};
/// \endcode

class MoeInitRoutingQuantOperation : public AclNNOperation {
public:
    explicit MoeInitRoutingQuantOperation(const std::string &name, MoeInitRoutingQuantParam param);
    ~MoeInitRoutingQuantOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    MoeInitRoutingQuantParam param_;
};

}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_MOE_INIT_ROUTING_QUANT_OPERATION_H