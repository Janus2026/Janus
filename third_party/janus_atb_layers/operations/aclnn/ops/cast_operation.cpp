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
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "operations/aclnn/core/acl_nn_operation.h"
#include "aclnnop/aclnn_cast.h"
#include "cast_operation.h"

namespace atb_speed {
namespace common {

CastOperation::CastOperation(const std::string &name, AclNNCastParam param)
    : AclNNOperation(name), param_(param) {}

CastOperation::~CastOperation()
{
    ATB_SPEED_LOG_DEBUG("CastOperation deconstructor");
    this->DestroyOperation();
}

constexpr int MAX_DIMENSION = 8;

atb::Status CastOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                      atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");

    if (inTensorDescs.at(0).shape.dimNum > MAX_DIMENSION) {
        ATB_SPEED_LOG_ERROR(opName_ << " tensor dimension exceeds limit");
        return atb::ERROR_INVALID_PARAM;
    }

    outTensorDescs.at(0).shape = inTensorDescs.at(0).shape;
    outTensorDescs.at(0).dtype = param_.dtype;
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;

    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return atb::NO_ERROR;
}

uint32_t CastOperation::GetInputNum() const { return NUM1; }

uint32_t CastOperation::GetOutputNum() const { return NUM1; }

int CastOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack start");
    
    int ret = 0;
    ret = CreateAclNNInTensorVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(opName_ << " CreateAclNNInTensorVariantPack failed");
        return ret;
    }

    ret = CreateAclNNOutTensorVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(opName_ << " CreateAclNNOutTensorVariantPack failed");
        return ret;
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack end");
    return atb::NO_ERROR;
}

atb::Status CastOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());

    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->tensorIdx = 0;
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = variantPack.inTensors.at(0);
    aclnnTensor->strides = GetCopyTensorStride(aclnnTensor->atbTensor.desc.shape);

    aclnnTensor->tensor = aclCreateTensor(
        aclnnTensor->atbTensor.desc.shape.dims, aclnnTensor->atbTensor.desc.shape.dimNum,
        aclnnTensor->atbTensor.desc.dtype, aclnnTensor->strides.data(), 0,
        aclnnTensor->atbTensor.desc.format, aclnnTensor->atbTensor.desc.shape.dims,
        aclnnTensor->atbTensor.desc.shape.dimNum, aclnnTensor->atbTensor.deviceData);

    if (aclnnTensor->tensor == nullptr) {
        ATB_SPEED_LOG_ERROR(opName_ << " Create input tensor failed");
        return atb::ERROR_INTERNAL_ERROR;
    }

    aclnnVariantPack.aclInTensors[0] = aclnnTensor;
    return atb::NO_ERROR;
}

atb::Status CastOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());

    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->tensorIdx = 0;
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = variantPack.outTensors.at(0);
    aclnnTensor->strides = GetCopyTensorStride(aclnnTensor->atbTensor.desc.shape);

    aclnnTensor->tensor = aclCreateTensor(
        aclnnTensor->atbTensor.desc.shape.dims, aclnnTensor->atbTensor.desc.shape.dimNum,
        param_.dtype, aclnnTensor->strides.data(), 0,
        aclnnTensor->atbTensor.desc.format, aclnnTensor->atbTensor.desc.shape.dims,
        aclnnTensor->atbTensor.desc.shape.dimNum, aclnnTensor->atbTensor.deviceData);

    if (aclnnTensor->tensor == nullptr) {
        ATB_SPEED_LOG_ERROR(opName_ << " Create output tensor failed");
        return atb::ERROR_INTERNAL_ERROR;
    }

    aclnnVariantPack.aclOutTensors[0] = aclnnTensor;
    return atb::NO_ERROR;
}

int CastOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;

    int ret = aclnnCastGetWorkspaceSize(
        aclnnVariantPack.aclInTensors[0]->tensor,
        param_.dtype,
        aclnnVariantPack.aclOutTensors[0]->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    if (ret != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR(opName_ << " GetWorkspaceSize failed with error code: " << ret);
        return ret;
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end");
    return atb::NO_ERROR;
}

int CastOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    int ret = aclnnCast(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    if (ret != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR(opName_ << " ExecuteAclNNOp failed");
    }
    return ret;
}

} // namespace common
} // namespace atb_speed