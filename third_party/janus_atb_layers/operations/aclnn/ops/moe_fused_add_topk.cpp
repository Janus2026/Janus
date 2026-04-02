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

#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "operations/aclnn/utils/utils.h"

#include "moe_fused_add_topk.h"
#include "aclnn_moe_fused_add_topk.h"



namespace atb_speed {
namespace common {

MoeFusedAddTopkOperation::MoeFusedAddTopkOperation(
    const std::string &name,
    AclNNMoeFusedAddTopkParam param) : AclNNOperation(name), param_(param) {}

MoeFusedAddTopkOperation::~MoeFusedAddTopkOperation()
{
    ATB_SPEED_LOG_DEBUG("MoeFusedAddTopkOperation deconstructor");
    this->DestroyOperation();
}

uint32_t MoeFusedAddTopkOperation::GetInputNum() const
{
    uint32_t inputNum = DIM2;
    if (param_.enableExpertMapping) {
        inputNum += DIM2;
    }
    return inputNum;
}

uint32_t MoeFusedAddTopkOperation::GetOutputNum() const
{
    return DIM2;
}

atb::Status MoeFusedAddTopkOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                        atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");

    outTensorDescs.at(DIM0).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(DIM0).dtype = aclDataType::ACL_FLOAT;
    outTensorDescs.at(DIM0).shape.dimNum = DIM2;

    outTensorDescs.at(DIM1).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(DIM1).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(DIM1).shape.dimNum = DIM2;
    
    outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
    outTensorDescs.at(DIM0).shape.dims[DIM1] = param_.k;
    outTensorDescs.at(DIM1).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
    outTensorDescs.at(DIM1).shape.dims[DIM1] = param_.k;
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return 0;
}


int MoeFusedAddTopkOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); i++) {
            
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(aclnnTensor->atbTensor.desc.shape);

        aclnnTensor->tensor = aclCreateTensor(
            aclnnTensor->atbTensor.desc.shape.dims, aclnnTensor->atbTensor.desc.shape.dimNum,
            aclnnTensor->atbTensor.desc.dtype, aclnnTensor->strides.data(), 0,
            aclnnTensor->atbTensor.desc.format, aclnnTensor->atbTensor.desc.shape.dims,
            aclnnTensor->atbTensor.desc.shape.dimNum, aclnnTensor->atbTensor.deviceData);

        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int MoeFusedAddTopkOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(aclnnTensor->atbTensor.desc.shape);
        aclnnTensor->tensor = aclCreateTensor(
            aclnnTensor->atbTensor.desc.shape.dims, aclnnTensor->atbTensor.desc.shape.dimNum,
            aclnnTensor->atbTensor.desc.dtype, aclnnTensor->strides.data(), 0,
            aclnnTensor->atbTensor.desc.format, aclnnTensor->atbTensor.desc.shape.dims,
            aclnnTensor->atbTensor.desc.shape.dimNum, aclnnTensor->atbTensor.deviceData);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " OutTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}


int MoeFusedAddTopkOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack start");
    int ret = 0;
    ret = CreateAclNNInTensorVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " AclNNTensor CreateAclNNInTensorVariantPack fail");
        return ret;
    }
    ret = CreateAclNNOutTensorVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " AclNNTensor CreateAclNNOutTensorVariantPack fail");
        return ret;
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack end");
    return atb::NO_ERROR;
}


int MoeFusedAddTopkOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;

    int ret = aclnnMoeFusedAddTopkGetWorkspaceSize(
            aclnnVariantPack.aclInTensors[DIM0]->tensor, 
            aclnnVariantPack.aclInTensors[DIM1]->tensor, 
            param_.enableExpertMapping ? aclnnVariantPack.aclInTensors[DIM2]->tensor : nullptr,
            param_.enableExpertMapping ? aclnnVariantPack.aclInTensors[DIM3]->tensor : nullptr,
            param_.groupNum, param_.groupTopk, param_.n, param_.k, param_.activationType, 
            param_.isNorm, param_.scale, param_.enableExpertMapping, 
            aclnnVariantPack.aclOutTensors.at(DIM0)->tensor, 
            aclnnVariantPack.aclOutTensors.at(DIM1)->tensor,
            &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);

    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}


int MoeFusedAddTopkOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddmm start");
    int ret = aclnnMoeFusedAddTopk(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddmm end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed

