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
#ifndef ATB_SPEED_PLUGIN_ACLNN_SCATTER_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_SCATTER_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {

enum class ReduceType {
    REPLACE = 0,
    ADD = 1,
    MULTIPLY = 2
};

struct AclNNScatterParam {
    int64_t dim = 0;
    ReduceType reduce = ReduceType::REPLACE;
};

class ScatterOperation : public AclNNOperation {
public:
    explicit ScatterOperation(const std::string &name, AclNNScatterParam param, bool isInplace);
    ~ScatterOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    int CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;

private:
    AclNNScatterParam param_;
    bool isInplace_;
};
} // namespace common
} // namespace atb_speed
#endif // ATB_SPEED_PLUGIN_ACLNN_SCATTER_OPERATION_H