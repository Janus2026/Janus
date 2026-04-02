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
#ifndef ATB_SPEED_PLUGIN_ACLNN_SWIGLU_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_SWIGLU_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {
/// This class defines a matrix operation combines the RmsNorm operator
/// reducing the operations of moving data in and out.
///
/// This class makes use of `aclnnRmsNormGetWorkspaceSize` and `aclnnRmsNorm` from the AscendCL API.
///
/// Operation's Inputs:
/// Name            | Dtype                       | Shape   |
/// ----------------|-----------------------------|---------|
/// x               | FLOAT, FLOAT16, BFLOAT16    |         |
/// dim             | int64                       | Scalar  |
///
/// Operations's Outputs:
/// Name    | Dtype                       | Shape   |
/// --------|-----------------------------|---------|
/// out     | FLOAT, FLOAT16, BFLOAT16    |         |
///
/// Example:
/// \code
/// enum InTensorIdx : uint32_t {
///     IN_INPUT1 = 0,
/// };
///
/// enum OutTensorIdx : uint32_t {
///     OUT1 = 0,
/// };
///
/// atb::Node swiGluNode;
/// swiGluNode.operation = new atb_speed::common::SwiGluOperation("SwiGlu",  dim=-1);
/// swiGluNode.outTensorIds = {OUT1};
/// swiGluNode.inTensorIds = {IN_INPUT1};
///
/// // Add the operation node to the graph as required
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(swiGluNode);
/// \endcode
class SwiGluOperation : public AclNNOperation {
public:
    explicit SwiGluOperation(const std::string &name, long dim=-1);
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    long dim;
    atb::Status CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    std::shared_ptr<AclNNTensor> CreateTensor(atb::Tensor atbTensor, int tensorIdx) const;

};
} // namespace common
} // namespace atb_speed
#endif
