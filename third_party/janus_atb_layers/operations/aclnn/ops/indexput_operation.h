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
#ifndef ATB_SPEED_PLUGIN_ACLNN_INDEXPUT_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_INDEXPUT_OPERATION_H
#include <iostream>
#include <sstream>
#include <string>

#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {
/// A struct defines `Indexput`'s parameter.
struct AclNNIndexputParam {
    /// A flag indicating whether accumulation or update.
    bool accumulate = false;
    /// Check whether the index is within the valid range flag.
    bool unsafe = true;

    std::string ToString() const
    {
        std::ostringstream oss;
        oss << "AclNNIndexputParam {" << std::endl;
        oss << "  accumulate: " << accumulate << std::endl;
        oss << "  unsafe: " << unsafe << std::endl;
        oss << "}";
        return oss.str();
    }
};

/// This class defines a matrix operation that supports
/// update or accumulate the data at the corresponding coordinates of the input x
/// with the input value according to the indices.
///
/// This class makes use of `aclnnIndexPutImplGetWorkspaceSize` and `aclnnIndexPutImpl` from the AscendCL API.
///
/// Operation's Inputs:
/// Name            | Dtype   | Shape |
/// ----------------|---------|-------|
/// input   | float32, float16, bfloat16 | The dimension is not greater than 8 |
/// indices | int32, int64, bool | [n] |
/// values  | same as input      | The dimension is the same as input. The first dimension is equal to the indices. |
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_INDICES,
///     IN_VALUES,
/// };
///
/// atb::Node indexPutNode;
/// AclNNIndexputParam indexPutParam;
/// indexPutParam.dim = 0;
/// indexPutNode.inTensorIds = {IN_INPUT, IN_INDICES, IN_VALUES};
/// indexPutNode.outTensorIds = {IN_INPUT};
/// indexPutNode.operation = new atb_speed::common::IndexSelectOperation("IndexPutNode", indexPutParam);
///
/// // Add the operation node to the graph as required
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(indexPutNode);
/// \endcode

class IndexputOperation : public AclNNOperation {
public:
    explicit IndexputOperation(const std::string &name, AclNNIndexputParam param);
    ~IndexputOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;

private:
    atb::Status CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    std::vector<aclTensor *> vectorList;
    AclNNIndexputParam param_;
};
} // namespace common
} // namespace atb_speed
#endif