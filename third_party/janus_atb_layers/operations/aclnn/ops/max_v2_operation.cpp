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
#include "max_v2_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_max_v2.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {

MaxV2Operation::MaxV2Operation(const std::string &name) : AclNNOperation(name) {}

MaxV2Operation::MaxV2Operation(const std::string &name, atb_speed::common::AclNNMaxV2Param param)
    : AclNNOperation(name), param_(param)
{
}

MaxV2Operation::~MaxV2Operation()
{
    ATB_SPEED_LOG_DEBUG("MaxV2Operation deconstruct");
    this->DestroyOperation();
}

uint32_t MaxV2Operation::GetInputNum() const { return NUM1; }

uint32_t MaxV2Operation::GetOutputNum() const { return NUM1; }

atb::Status MaxV2Operation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                                       atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MaxV2Operation infer shape start");
    outTensorDesc.at(0).format = inTensorDesc.at(0).format;
    outTensorDesc.at(0).dtype = inTensorDesc.at(0).dtype;
    uint32_t inputDimNum = inTensorDesc.at(0).shape.dimNum;
    uint32_t outputDimNum = inputDimNum;
    uint32_t realDim = this->param_.dims[0] < 0 ? this->param_.dims[0] + inputDimNum : this->param_.dims[0];

    if (!param_.keepdim) {
        outputDimNum -= 1;
    }
    outTensorDesc.at(0).shape.dimNum = outputDimNum;

    uint32_t j = 0;
    for (uint32_t i = 0; i < outputDimNum; ++i) {
        if (i == realDim && param_.keepdim) {
            outTensorDesc.at(0).shape.dims[i] = 1;
            j++;
        } else {
            outTensorDesc.at(0).shape.dims[j++] = inTensorDesc.at(0).shape.dims[i];
        }
    }

    ATB_SPEED_LOG_DEBUG(opName_ << "MaxV2Operation InferShape end");

    return atb::NO_ERROR;
}

int MaxV2Operation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
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

int MaxV2Operation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
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


int MaxV2Operation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
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

std::shared_ptr<AclNNTensor> MaxV2Operation::CreateTensor(atb::Tensor atbTensor, int tensorIdx) const
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

int MaxV2Operation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclIntArray* dims = aclCreateIntArray(this->param_.dims.data(), this->param_.dims.size());
    int ret = aclnnMaxV2GetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor, dims,
                                         this->param_.keepdim, false, aclnnVariantPack.aclOutTensors.at(0)->tensor,
                                         &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                    << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                    << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int MaxV2Operation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnMaxV2(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end, ret:" << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed