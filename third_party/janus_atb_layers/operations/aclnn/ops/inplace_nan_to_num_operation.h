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

#ifndef ATB_SPEED_PLUGIN_ACLNN_INPLACE_NAN_TO_NUM_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_INPLACE_NAN_TO_NUM_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"


namespace atb_speed::common {


struct AclNNNanToNumParam {
    /// nanValue: Input parameter that replaces NaN values in tensor elements. The data type supports FLOAT.
    /// posInfValue: Input parameter that replaces positive infinity values in tensor elements.
    ///              The data type supports FLOAT.
    /// negInfValue: Input parameter that replaces negative infinity values in tensor elements.
    ///              The data type supports FLOAT.
    float nanValue = 0.0;
    float posInfValue = 65504.0;
    float negInfValue = -65504.0;
};

/// Replace NaN, positive infinity, and negative infinity values in the input with the
/// values specified by nan, posinf, and neginf, respectively.
///
/// Operation's Inputs: \n
/// | Name   | Dtype                    | Shape     | \n
/// |--------|--------------------------|-----------| \n
/// | x      | FLOAT16,FLOAT32,INT8,INT16,INT32,INT64,UINT8,BOOL,BFLOAT16 | [-1,…,-1] | \n
///
/// Operation's Outputs: it is inplace replace.\n
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
/// };
///
/// atb::GraphParam opGraph;
/// atb::Node nanToNumNode;
/// atb_speed::common::AclNNNanToNumParam NanToNumParam;
/// NanToNumParam.posInfValue = 50000.0;
/// NanToNumParam.negInfValue = -50000.0;
/// nanToNumNode.operation = new atb_speed::common::InplaceNanToNumOperation("nanToNumNode", NanToNumParam);
/// nanToNumNode.inTensorIds = { IN_INPUT };
/// nanToNumNode.outTensorIds = { IN_INPUT };
/// opGraph.nodes.push_back(nanToNumNode);
///
/// \endcode
class InplaceNanToNumOperation : public AclNNOperation {
public:
    explicit InplaceNanToNumOperation(const std::string &name, AclNNNanToNumParam param);
    ~InplaceNanToNumOperation() override;
    atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDesc,
        atb::SVector<atb::TensorDesc> &outTensorDesc
    ) const override;
    [[nodiscard]] uint32_t GetInputNum() const override;
    [[nodiscard]] uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;

private:
    AclNNNanToNumParam param_;
    std::string opName_;
};
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_INPLACE_NAN_TO_NUM_OPERATION_H

