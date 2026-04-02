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
#include "split_with_size_operation.h"

namespace atb_speed {
namespace common {

SplitWithSizeOperation::SplitWithSizeOperation(
    const std::string &name,
    AclNNSplitWithSizeParam param) : AclNNOperation(name), param_(param)
{
    outputTensorVector.resize(param.num);
}

SplitWithSizeOperation::~SplitWithSizeOperation() {
}

atb::Status SplitWithSizeOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "SplitWithSizeOperation infer shape start");
    int splitSize = inTensorDescs.at(DIM0).shape.dims[param_.dim] / param_.num;
    int remainSize = inTensorDescs.at(DIM0).shape.dims[param_.dim] % param_.num;
    for (size_t i = 0; i < GetOutputNum(); i++) {
        outTensorDescs.at(i) = inTensorDescs.at(DIM0);
        if (i < static_cast<size_t>(remainSize)) {
            outTensorDescs.at(i).shape.dims[param_.dim] = splitSize + 1;
        } else {
            outTensorDescs.at(i).shape.dims[param_.dim] = splitSize;
        }
    }
    ATB_SPEED_LOG_DEBUG(opName_ << "SplitWithSizeOperation infer shape end");
    return 0;
}

uint32_t SplitWithSizeOperation::GetInputNum() const
{
    // 1: aclInTensors size
    return 1;
}

uint32_t SplitWithSizeOperation::GetOutputNum() const
{
    return param_.num;
}

int SplitWithSizeOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
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

int SplitWithSizeOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); i++) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(variantPack.inTensors.at(i));
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        aclnnTensor->tensor = aclCreateTensor(
            squeezedAtbTensor.desc.shape.dims, squeezedAtbTensor.desc.shape.dimNum,
            squeezedAtbTensor.desc.dtype, aclnnTensor->strides.data(), 0,
            squeezedAtbTensor.desc.format, squeezedAtbTensor.desc.shape.dims,
            squeezedAtbTensor.desc.shape.dimNum, squeezedAtbTensor.deviceData);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int SplitWithSizeOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(variantPack.outTensors.at(i));
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        aclnnTensor->tensor = aclCreateTensor(
            squeezedAtbTensor.desc.shape.dims, squeezedAtbTensor.desc.shape.dimNum,
            squeezedAtbTensor.desc.dtype, aclnnTensor->strides.data(), 0,
            squeezedAtbTensor.desc.format, squeezedAtbTensor.desc.shape.dims,
            squeezedAtbTensor.desc.shape.dimNum, squeezedAtbTensor.deviceData);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " OutTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); i++) {
        outputTensorVector[i] = aclnnVariantPack.aclOutTensors.at(i)->tensor;
    }
    return atb::NO_ERROR;
}

int SplitWithSizeOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclTensorList *out = aclCreateTensorList(outputTensorVector.data(), outputTensorVector.size());

    std::vector<int64_t> splitSizeVec;
    int splitSize = aclnnVariantPack.aclInTensors.at(DIM0)->atbTensor.desc.shape.dims[param_.dim] / param_.num;
    int remainSize = aclnnVariantPack.aclInTensors.at(DIM0)->atbTensor.desc.shape.dims[param_.dim] % param_.num;
    for (size_t i = 0; i < GetOutputNum(); i++) {
        if (i < static_cast<size_t>(remainSize)) {
            splitSizeVec.emplace_back(splitSize + 1);
        } else {
            splitSizeVec.emplace_back(splitSize);
        }
    }
    aclIntArray *splitSizeIntArray = aclCreateIntArray(splitSizeVec.data(), splitSizeVec.size());

    int ret = aclnnSplitWithSizeGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(DIM0)->tensor, splitSizeIntArray, param_.dim, out,
        &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);

    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int SplitWithSizeOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnSplitWithSize start");
    int ret = aclnnSplitWithSize(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnSplitWithSize end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed