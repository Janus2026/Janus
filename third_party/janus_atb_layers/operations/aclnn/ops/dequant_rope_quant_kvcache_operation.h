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
#ifndef ATB_SPEED_PLUGIN_ACLNN_DEQUANT_ROPE_QUANT_KVCACHE_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_DEQUANT_ROPE_QUANT_KVCACHE_OPERATION_H
#include "acl/acl.h"
#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {
struct AclNNDequantRopeQuantKvcacheParam {
    std::vector<int64_t> sizeSpilts = {128 * 8, 128, 128};
    bool kvOutput = true;
    std::string quantMode = "static";
    std::string layout = "BSND";
    bool enableDequant = false;
};

class DequantRopeQuantKvcacheOperation : public AclNNOperation {
public:
    explicit DequantRopeQuantKvcacheOperation(const std::string &name, AclNNDequantRopeQuantKvcacheParam param);
    ~DequantRopeQuantKvcacheOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    int CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
    int CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    int CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;

private:
    AclNNDequantRopeQuantKvcacheParam param_;
};
} // namespace common
} // namespace atb_speed
#endif
