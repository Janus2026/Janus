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
#include <securec.h>
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "aclnnop/aclnn_prompt_flash_attention_v3.h"

#include "prompt_flash_attention_operation.h"

namespace atb_speed {
namespace common {
PromptFlashAttentionOperation::PromptFlashAttentionOperation(const std::string &name,
    AclNNFlashAttentionParam param)
    : AclNNOperation(name), param_(param)
{
    ATB_SPEED_LOG_DEBUG("PromptFlashAttentionOperation, param: " << param_.ToString());
}

PromptFlashAttentionOperation::~PromptFlashAttentionOperation()
{
    ATB_SPEED_LOG_DEBUG("~PromptFlashAttentionOperation");
}

uint32_t PromptFlashAttentionOperation::GetInputNum() const
{
    return param_.needMask ? NUM6 : NUM5;
}

uint32_t PromptFlashAttentionOperation::GetOutputNum() const
{
    return NUM1;
}

atb::Status PromptFlashAttentionOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
    outTensorDescs.at(0) = inTensorDescs.at(0);
    // if input layout is BNSD_BSND, input shape is BNSD and then output shape is BSND;
    // otherwise, output shape equals to input shape.
    if (param_.inputLayout == "BNSD_BSND") {
        outTensorDescs.at(0).shape.dims[DIM1] = inTensorDescs.at(0).shape.dims[DIM2];
        outTensorDescs.at(0).shape.dims[DIM2] = inTensorDescs.at(0).shape.dims[DIM1];
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return 0;
}

int PromptFlashAttentionOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
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

int PromptFlashAttentionOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    uint32_t inputNum = GetInputNum();
    aclnnVariantPack.aclInTensors.resize(inputNum);
    int inTensorIdx = 0;
    for (size_t i = 0; i < inputNum; i++) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        if (i == 3) { // 3 is empty tensor, skip
            inTensorIdx++;
        }
        aclnnTensor->tensorIdx = inTensorIdx++;
        if (i == inputNum - 2) { // qSeqLens is 2nd last input tensor
            aclnnTensor->needUpdateTensorDataPtr = false;
            ConvertTensorToSeqLengths(aclnnTensor->atbTensor, actualSeqLengths_);
        } else if (i == inputNum - 1) { // kvSeqLens is the last input tensor
            aclnnTensor->needUpdateTensorDataPtr = false;
            ConvertTensorToSeqLengths(aclnnTensor->atbTensor, actualSeqLengthsKv_);
        } else {
            aclnnTensor->needUpdateTensorDataPtr = true;
            atb::Tensor atbTensor = variantPack.inTensors.at(i);
            aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
            aclnnTensor->tensor = aclCreateTensor(atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum,
                atbTensor.desc.dtype, aclnnTensor->strides.data(), 0, atbTensor.desc.format, atbTensor.desc.shape.dims,
                atbTensor.desc.shape.dimNum, atbTensor.deviceData);
            if (aclnnTensor->tensor == nullptr) {
                ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor index " << i << " create fail");
                return atb::ERROR_INTERNAL_ERROR;
            }
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int PromptFlashAttentionOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); i++) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(variantPack.outTensors.at(i));
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        aclnnTensor->tensor = aclCreateTensor(squeezedAtbTensor.desc.shape.dims, squeezedAtbTensor.desc.shape.dimNum,
            squeezedAtbTensor.desc.dtype, aclnnTensor->strides.data(), 0, squeezedAtbTensor.desc.format,
            squeezedAtbTensor.desc.shape.dims, squeezedAtbTensor.desc.shape.dimNum, squeezedAtbTensor.deviceData);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " OutTensor index " << i << " create fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int PromptFlashAttentionOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << "GetWorkspaceSize start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclTensor *query = aclnnVariantPack.aclInTensors.at(0)->tensor;
    aclTensor *key = aclnnVariantPack.aclInTensors.at(1)->tensor;
    aclTensor *value = aclnnVariantPack.aclInTensors.at(2)->tensor;
    aclTensor *pseShift = nullptr;
    aclTensor *attenMask = param_.needMask ? aclnnVariantPack.aclInTensors.at(3)->tensor : nullptr;

    int ret = aclnnPromptFlashAttentionV3GetWorkspaceSize(query, key, value, pseShift, attenMask, actualSeqLengths_,
        actualSeqLengthsKv_, nullptr, nullptr, nullptr, nullptr, nullptr, param_.numHeads, param_.scaleValue,
        param_.preTokens, param_.nextTokens, const_cast<char *>(param_.inputLayout.c_str()), param_.numKeyValueHeads,
        param_.sparseMode, param_.innerPrecise, aclnnVariantPack.aclOutTensors.at(0)->tensor,
        &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);
        
    ATB_SPEED_LOG_DEBUG(opName_ << " end, ret:" << ret << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize <<
        ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int PromptFlashAttentionOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    return aclnnPromptFlashAttentionV3(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor,
        stream);
}
} // namespace common
} // namespace atb_speed
