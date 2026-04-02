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
#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_GROUPED_MATMUL_SWIGLU_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_GROUPED_MATMUL_SWIGLU_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

struct AclNNMoeGroupedSwigluMatmulParam {
    bool transposeB = false;  /// A flag indicating wheter the second input matrix needs to be transposed
    int quantType = 0;  /// The quantization type of the operation
    bool hasBias = false;  /// A flag indicating whether the matmul operation includes a bias tensor
    aclDataType outDataType = ACL_FLOAT16;  /// The data type of the outpuot of the oepration
};

class MoeGroupedMatmulSwigluQuantOperation : public AclNNOperation {
public:
   explicit MoeGroupedMatmulSwigluQuantOperation(const std::string &name, AclNNMoeGroupedSwigluMatmulParam param);
    ~MoeGroupedMatmulSwigluQuantOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
    AclNNMoeGroupedSwigluMatmulParam param_;
    static constexpr uint32_t INPUT_NUM = 5U;
    static constexpr uint32_t OUTPUT_NUM = 2U;
};
}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_MOE_GROUPED_MATMUL_OPERATION_H

