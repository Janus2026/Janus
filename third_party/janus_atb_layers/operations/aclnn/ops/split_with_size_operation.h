/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#ifndef ATB_SPEED_PLUGIN_ACLNN_SPLIT_WITH_SIZE_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_SPLIT_WITH_SIZE_OPERATION_H
#include "aclnnop/aclnn_split_with_size.h"
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "operations/aclnn/core/acl_nn_tensor.h"


namespace atb_speed {
namespace common {

struct AclNNSplitWithSizeParam {
    int64_t dim = 0;
    uint64_t num = 1;
};

class SplitWithSizeOperation : public AclNNOperation {
public:
    explicit SplitWithSizeOperation(const std::string &name, AclNNSplitWithSizeParam param);
    ~SplitWithSizeOperation() override;
    uint32_t GetInputNum() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    int CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
    int CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    AclNNSplitWithSizeParam param_;
    std::vector<aclTensor *> outputTensorVector;
};
}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_SPLIT_WITH_SIZE_OPERATION_Hs