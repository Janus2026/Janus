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

#ifndef ATB_SPEED_PLUGIN_ACLNN_MASKEDFILL_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MASKEDFILL_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed::common {
struct InplaceMaskedFillTensorParam {
    float value = 0;
    aclDataType outDataType = ACL_FLOAT16;
};

/// This class defines an operator that replaces the value in the tensor with another specified value.
///
/// This class makes uses of `aclnnInplaceMaskedFillScalarGetWorkspaceSize` and `aclnnInplaceMaskedFillScalar`
/// form the AscendCL API.
///
/// Inputs to the operator
/// Name         | Dtype               | Shape |
/// -------------|---------------------|-------|
/// input        | float16 or bfloat16 | [m]   |
///
/// Outputs of the operator:
/// Name         | Dtype               | Shape |
/// -------------|---------------------|-------|
/// output       | float16 or bfloat16 | [m]   |
/// Note: The output is a placeholder that wouldn't be written during executing.
///
/// Example:
/// \code
/// enum InTensorIdx : uint32_t {INPUT = 0};
///
/// enum OutTensorIdx : uint32_t {OUT = 0};
///
/// atb::Node &maskedFillNode = opGraph.nodes.at(nodeId++);
/// atb_speed::common::InplaceMaskedFillTensorParam fillParam;
/// fillParam.value = param.fillValue;
/// fillParam.outDataType = param.outDataType;
/// maskedFillNode.operation = new atb_speed::common::InplaceMaskedFillTensorOperation("MaskedFill", fillParam);
/// maskedFillNode.inTensorIds = {INPUT};
/// maskedFillNode.outTensorIds = {OUTPUT};
/// \endcode

class InplaceMaskedFillTensorOperation : public AclNNOperation {
public:
    explicit InplaceMaskedFillTensorOperation(const std::string &name, InplaceMaskedFillTensorParam param);
    ~InplaceMaskedFillTensorOperation() override;
    atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs
    ) const override;
    [[nodiscard]] uint32_t GetInputNum() const override;
    [[nodiscard]] uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;

private:
    InplaceMaskedFillTensorParam param_;
    std::string opName_;
};
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_MASKEDFILL_OPERATION_H