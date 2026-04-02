/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#include "len_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_range.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {

LenOperation::LenOperation(const std::string &name) : AclNNOperation(name) {}

LenOperation::~LenOperation()
{
    ATB_SPEED_LOG_DEBUG("LenOperation deconstruct");
    this->DestroyOperation();
}

uint32_t LenOperation::GetInputNum() const { return NUM1; }

uint32_t LenOperation::GetOutputNum() const { return NUM1; }

atb::Status LenOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDesc,
    atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " LenOperation infer shape start");
    outTensorDesc.at(0).format = inTensorDesc.at(0).format;
    outTensorDesc.at(0).dtype = aclDataType::ACL_INT32;
    outTensorDesc.at(0).shape.dimNum = 1;
    outTensorDesc.at(0).shape.dims[0] = 1;
    ATB_SPEED_LOG_DEBUG(opName_ << "LenOperation InferShape end");

    return atb::NO_ERROR;
}

int LenOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
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

int LenOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
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


int LenOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
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

std::shared_ptr<AclNNTensor> LenOperation::CreateTensor(atb::Tensor atbTensor, int tensorIdx) const
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

int LenOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    auto start = aclnnVariantPack.aclInTensors.at(DIM0)->atbTensor.desc.shape.dims[DIM0];
    auto end = start + 1;
    auto step = 1;

    int ret = aclnnRangeGetWorkspaceSize(
        aclCreateScalar(&start, aclDataType::ACL_INT32),
        aclCreateScalar(&end, aclDataType::ACL_INT32),
        aclCreateScalar(&step, aclDataType::ACL_INT32),
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
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

int LenOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnRange(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end, ret:" << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed