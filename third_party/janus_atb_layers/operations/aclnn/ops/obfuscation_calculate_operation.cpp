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
#include "acl/acl.h"
#include "aclnnop/aclnn_obfuscation_calculate.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "obfuscation_calculate_operation.h"

namespace atb_speed {
namespace common {

ObfuscationCalculateOperation::ObfuscationCalculateOperation(
    const std::string &name, ObfuscationCalculateParam param) : AclNNOperation(name), param_(param) {}

ObfuscationCalculateOperation:: ~ObfuscationCalculateOperation()
{
    ATB_SPEED_LOG_DEBUG("ObfuscationCalculateOperation deconstructor");
    this->DestroyOperation();
}

atb::Status ObfuscationCalculateOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG("ObfuscationCalculateOperation infer shape start");
    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = inTensorDescs.at(DIM0).dtype;
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;
    for (uint32_t i = 0; i < inTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(DIM0).shape.dims[i] = inTensorDescs.at(DIM0).shape.dims[i];
    }
    ATB_SPEED_LOG_DEBUG("ObfuscationCalculateOperation infer shape end");
    return 0;
}

uint32_t ObfuscationCalculateOperation::GetInputNum() const { return NUM1; }

uint32_t ObfuscationCalculateOperation::GetOutputNum() const { return NUM1; }

int ObfuscationCalculateOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    uint32_t inputNum = GetInputNum();
    aclnnVariantPack.aclInTensors.resize(inputNum);
    atb::Tensor atbTensor = variantPack.inTensors.at(0);
    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = atbTensor;
    aclnnTensor->tensorIdx = 1;
    aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
    aclnnTensor->tensor = aclCreateTensor(
        atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum,
        atbTensor.desc.dtype, aclnnTensor->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims,
        atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    aclnnVariantPack.aclInTensors.at(0) = aclnnTensor;
    return atb::NO_ERROR;
}

int ObfuscationCalculateOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationCalculateGetWorkspaceSize start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnObfuscationCalculateGetWorkspaceSize(
        param_.fd,
        aclnnVariantPack.aclInTensors.at(0)->tensor,
        param_.hiddenSizePerRank,
        param_.cmd,
        aclnnVariantPack.aclOutTensors.at(0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationCalculateGetWorkspaceSize end, ret:" <<
        ret << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize <<
        ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);

    return ret;
}

int ObfuscationCalculateOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationCalculate start");
    int ret = aclnnObfuscationCalculate(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationCalculate end, ret:" << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed