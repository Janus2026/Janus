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

#ifndef ATB_SPEED_PLUGIN_ACLNN_INDEX_SELECT_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_INDEX_SELECT_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"


namespace atb_speed::common {

/// A struct defines `IndexSelect`'s parameter.
struct IndexSelectParam {
    /// A flag indicating the specified dimension of the input tensor,
    /// the range is [-input.dim(), input.dim() - 1].
    int64_t dim = 0;
};

/// This class defines a matrix operation that supports
/// extract elements from the specified dimension dim of the input Tensor according to the index sequence numbers
/// and save them to the out Tensor.
///
/// This class makes use of `aclnnIndexSelectGetWorkspaceSize` and `aclnnIndexSelect` from the AscendCL API.
///
/// Operation's Inputs:
/// Name            | Dtype   | Shape |
/// ----------------|---------|-------|
/// input | float32, float16, bfloat16 | The dimension is not greater than 8 |
/// index | int32, int64               | [n] |
///
/// Operations's Outputs:
/// Name   | Dtype               | Shape |
/// -------|---------------------|-------|
/// output | same as input | The dimension is the same as input. The length of the dim dimension is equal to the index.|
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_INDEX,
///     OUT,
/// };
///
/// atb::Node indexSelectNode;
/// IndexSelectParam indexSelectParam;
/// indexSelectParam.dim = 0;
/// indexSelectNode.inTensorIds = {IN_INPUT, IN_INDEX};
/// indexSelectNode.outTensorIds = {OUT};
/// indexSelectNode.operation = new atb_speed::common::IndexSelectOperation("IndexSelectNode", IndexSelectParam);
///
/// // Add the operation node to the graph as required
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(indexSelectNode);
/// \endcode

class IndexSelectOperation : public AclNNOperation {
public:
    explicit IndexSelectOperation(const std::string &name, IndexSelectParam param);
    ~IndexSelectOperation() override;
    atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs
    ) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    IndexSelectParam param_;
    std::string opName_;
};
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_INDEX_SELECT_OPERATION_H
