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

#include "inplace_nan_to_num_operation.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_nan_to_num.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed::common {

InplaceNanToNumOperation::InplaceNanToNumOperation(
    const std::string &name, atb_speed::common::AclNNNanToNumParam param) : AclNNOperation(name), param_(param)
{
    this->opName_ = name;
    this->param_ = param;
}

InplaceNanToNumOperation::~InplaceNanToNumOperation()
{
    ATB_SPEED_LOG_DEBUG("InplaceNanToNumOperation deconstruct");
    this->DestroyOperation();
}

/**
    *
    * @param[in] inTensorDesc: FA: [batchSize, seqLen, hiddenSize]; PA: [seqLen, hiddenSize]
    * @param[in] outTensorDesc: FA: [batchSize, seqLen, hiddenSize]; PA: [seqLen, hiddenSize]
    * @return atb::Status
*/
atb::Status InplaceNanToNumOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDesc, atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "InplaceNanToNumOperation infer shape start");
    outTensorDesc.at(0).format = inTensorDesc.at(0).format;
    outTensorDesc.at(0).dtype = inTensorDesc.at(0).dtype;
    outTensorDesc.at(0).shape.dimNum = inTensorDesc.at(0).shape.dimNum;
    for (uint64_t i = 0; i < inTensorDesc.at(0).shape.dimNum; ++i) {
        outTensorDesc.at(0).shape.dims[i] = inTensorDesc.at(0).shape.dims[i];
    }

    ATB_SPEED_LOG_DEBUG(opName_ << "InplaceNanToNumOperation infer shape end"
        << " format: " << inTensorDesc.at(0).format << " dimNum: " << inTensorDesc.at(0).shape.dimNum
        << " dims: " << inTensorDesc.at(0).shape.dims[0]);
    return atb::NO_ERROR;
}

uint32_t InplaceNanToNumOperation::GetInputNum() const
{
    return NUM1;  // inputTensorNum = 1
}

uint32_t InplaceNanToNumOperation::GetOutputNum() const
{
    return NUM1;  // outputTensorNum = 1
}


atb::Status InplaceNanToNumOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
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

int InplaceNanToNumOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(
        opName_ << " SetAclNNWorkspaceExecutor start, nanValue: " <<
        param_.nanValue << " posInfValue: " <<
        param_.posInfValue << " negInfValue: " <<
        param_.negInfValue
   );
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnInplaceNanToNumGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor,  // self
        param_.nanValue,
        param_.posInfValue,
        param_.negInfValue,
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

int InplaceNanToNumOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnInplaceNanToNum(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end, ret: " << ret);
    return ret;
}

}  // namespace atb_speed::common

