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
#ifndef ATB_SPEED_PLUGIN_ACLNN_OBFUSCATION_SETUP_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_OBFUSCATION_SETUP_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {

struct ObfuscationSetupParam {
    int32_t fdtoClose = 0;
    int32_t dataType = 1; // 0: float32; 1: float16; 27: bfloat16
    int32_t hiddenSizePerRank = 1;
    int32_t tpRank = 0;
    int32_t cmd = 1; // 1: Normal mode; 3: Exit mode
    int32_t threadNum = 6; // thread num in aicpu
};

class ObfuscationSetupOperation : public AclNNOperation {
public:
    explicit ObfuscationSetupOperation(const std::string &name, ObfuscationSetupParam param);
    ~ObfuscationSetupOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    ObfuscationSetupParam param_;
};
} // namespace common
} // namespace atb_speed
#endif