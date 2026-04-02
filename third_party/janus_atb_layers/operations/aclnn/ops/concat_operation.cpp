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
#include "concat_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_cat.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {

ConcatOperation::ConcatOperation(const std::string &name, atb_speed::common::AclNNConcatParam param)
    : AclNNOperation(name), param_(param) {}

ConcatOperation::~ConcatOperation()
{
    ATB_SPEED_LOG_DEBUG("ConcatOperation deconstruct");
    this->DestroyOperation();
}

uint32_t ConcatOperation::GetInputNum() const { return NUM2; }

uint32_t ConcatOperation::GetOutputNum() const { return NUM1; }

atb::Status ConcatOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                        atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ConcatOperation infer shape start");
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape = inTensorDescs.at(0).shape;
    outTensorDescs.at(0).shape.dims[this->param_.dim] = inTensorDescs.at(0).shape.dims[this->param_.dim] + \
        inTensorDescs.at(1).shape.dims[this->param_.dim];
    ATB_SPEED_LOG_DEBUG(opName_ << "ConcatOperation InferShape end");
    return atb::NO_ERROR;
}

int ConcatOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
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

int ConcatOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
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

int ConcatOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
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

std::shared_ptr<AclNNTensor> ConcatOperation::CreateTensor(atb::Tensor atbTensor, int tensorIdx) const
{
    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = atbTensor;
    aclnnTensor->tensorIdx = tensorIdx;
    aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
    aclnnTensor->tensor = aclCreateTensor(atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum,
        atbTensor.desc.dtype, aclnnTensor->strides.data(),
        0, atbTensor.desc.format, atbTensor.desc.shape.dims,
        atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    return aclnnTensor;
}

int ConcatOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    std::vector<aclTensor *> tmp{aclnnVariantPack.aclInTensors.at(0)->tensor, \
                                    aclnnVariantPack.aclInTensors.at(1)->tensor};
    aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
    int ret = aclnnCatGetWorkspaceSize(tensorList, this->param_.dim,
        aclnnVariantPack.aclOutTensors.at(0)->tensor,
        &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret: " << ret
        << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
        << ", aclExecutor address: " << &this->aclnnOpCache_->aclExecutor
        << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int ConcatOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnCat(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end, ret:" << ret);
    return ret;
}

}
}