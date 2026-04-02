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

#ifndef ATB_SPEED_PLUGIN_ACLNN_MAX_V2_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MAX_V2_OPERATION_H

#include <vector>
#include "aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {
struct AclNNMaxV2Param {
    std::vector<int64_t> dims = {-1};
    bool keepdim = false;
};
class MaxV2Operation : public AclNNOperation {
public:
    explicit MaxV2Operation(const std::string &name);
    explicit MaxV2Operation(const std::string &name, AclNNMaxV2Param param);
    ~MaxV2Operation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                           atb::SVector<atb::TensorDesc> &outTensorDesc) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    int CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    int CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    int CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
    std::shared_ptr<AclNNTensor> CreateTensor(atb::Tensor atbTensor, int tensorIdx) const;
private:
    AclNNMaxV2Param param_;
};
} // namespace common
} // namespace atb_speed

#endif // ATB_SPEED_PLUGIN_ACLNN_MAX_V2_OPERATION_H
