/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#include "minimum_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_minimum.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {

MinimumOperation::MinimumOperation(const std::string &name) : AclNNOperation(name) {}

MinimumOperation::~MinimumOperation()
{
    ATB_SPEED_LOG_DEBUG("MinimumOperation deconstruct");
    this->DestroyOperation();
}

uint32_t MinimumOperation::GetInputNum() const { return NUM2; }

uint32_t MinimumOperation::GetOutputNum() const { return NUM1; }

atb::Status MinimumOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDesc,
    atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MinimumOperation infer shape start");
    outTensorDesc.at(0).format = inTensorDesc.at(0).format;
    outTensorDesc.at(0).dtype = inTensorDesc.at(0).dtype;
    outTensorDesc.at(0).shape.dimNum = inTensorDesc.at(0).shape.dimNum;
    for (uint64_t i = 0; i < inTensorDesc.at(0).shape.dimNum; ++i) {
        outTensorDesc.at(0).shape.dims[i] = inTensorDesc.at(0).shape.dims[i];
    }
    ATB_SPEED_LOG_DEBUG(opName_ << "MinimumOperation InferShape end");

    return atb::NO_ERROR;
}

int MinimumOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack start");
    int ret;

    ret = CreateAclNNInTensorVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " AclNNTensor CreateAclNNInTensorVariantPack fail");
        return ret;
    }

    ret = CreateAclNNOutTensorVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " AclNNTensor CreateAclNNOutTensorVariantPack fail");
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack end");
    return atb::NO_ERROR;
}

int MinimumOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.inTensors.at(i), i);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}


int MinimumOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclNnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclNnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclNnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.outTensors.at(i), i);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " outTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclNnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

std::shared_ptr<AclNNTensor> MinimumOperation::CreateTensor(atb::Tensor atbTensor, int tensorIdx) const
{
    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = atbTensor;
    aclnnTensor->tensorIdx = tensorIdx;
    aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
    aclnnTensor->tensor = aclCreateTensor(
        atbTensor.desc.shape.dims,
        atbTensor.desc.shape.dimNum,
        atbTensor.desc.dtype,
        aclnnTensor->strides.data(),
        0,
        atbTensor.desc.format,
        atbTensor.desc.shape.dims,
        atbTensor.desc.shape.dimNum,
        atbTensor.deviceData);
    return aclnnTensor;
}

int MinimumOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnMinimumGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor,  // self
        aclnnVariantPack.aclInTensors.at(1)->tensor,  // other
        aclnnVariantPack.aclOutTensors.at(0)->tensor,  // out
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(
        opName_ << " SetAclNNWorkspaceExecutor end"
                << ", ret: " << ret
                << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor
    );
    return ret;
}

int MinimumOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnMinimum(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end, ret:" << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed