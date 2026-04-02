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
#include "aclnnop/aclnn_scatter.h"
#include "scatter_operation.h"

namespace atb_speed {
namespace common {

ScatterOperation::ScatterOperation(
    const std::string &name,
    AclNNScatterParam param, bool isInplace) : AclNNOperation(name), param_(param), isInplace_(isInplace) {}

ScatterOperation::~ScatterOperation()
{
    ATB_SPEED_LOG_DEBUG("ScatterOperation deconstructor");
    this->DestroyOperation();
}

constexpr int MAX_DIMENSION = 8;

atb::Status ScatterOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                         atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");

    if (inTensorDescs.at(0).shape.dimNum > MAX_DIMENSION) {
        ATB_SPEED_LOG_ERROR(opName_ << " self tensor dim num exceeds limit");
        return atb::ERROR_INVALID_PARAM;
    }

    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;

    for (uint32_t i = 0; i < inTensorDescs.at(0).shape.dimNum; i++) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return atb::NO_ERROR;
}

uint32_t ScatterOperation::GetInputNum() const { return 3; }

uint32_t ScatterOperation::GetOutputNum() const { return 1; }

int ScatterOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack start");
    
    int ret = CreateAclNNInTensorVariantPack(variantPack);
    if (ret != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR(opName_ << " CreateAclNNInTensorVariantPack failed");
        return ret;
    }

    ret = CreateAclNNOutTensorVariantPack(variantPack);
    if (ret != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR(opName_ << " CreateAclNNOutTensorVariantPack failed");
        return ret;
    }

    aclnnOpCache_->executorRepeatable = false;

    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack end");
    return atb::NO_ERROR;
}

atb::Status ScatterOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());

    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) { // self, index, src
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
            ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " failed");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

atb::Status ScatterOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
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
        aclnnTensor->atbTensor.desc.dtype, aclnnTensor->strides.data(), 0,
        aclnnTensor->atbTensor.desc.format, aclnnTensor->atbTensor.desc.shape.dims,
        aclnnTensor->atbTensor.desc.shape.dimNum, aclnnTensor->atbTensor.deviceData);

    if (aclnnTensor->tensor == nullptr) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " OutTensor aclCreateTensor failed");
        return atb::ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack.aclOutTensors[0] = aclnnTensor;
    return atb::NO_ERROR;
}

int ScatterOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;

    int64_t reduceType = static_cast<int64_t>(param_.reduce);
    if (!isInplace_) {
        int ret = aclnnScatterGetWorkspaceSize(
            aclnnVariantPack.aclInTensors[0]->tensor, // self
            param_.dim, // scatter dim
            aclnnVariantPack.aclInTensors[1]->tensor, // index
            aclnnVariantPack.aclInTensors[2]->tensor, // src
            reduceType, // reduce type
            aclnnVariantPack.aclOutTensors[0]->tensor, // out
            &this->aclnnOpCache_->workspaceSize,
            &this->aclnnOpCache_->aclExecutor);
        if (ret != atb::NO_ERROR) {
            ATB_SPEED_LOG_ERROR(opName_ << " aclnnScatterGetWorkspaceSize failed with error code: " << ret);
            return ret;
        }
        ATB_SPEED_LOG_DEBUG(opName_ << " aclnnScatter SetAclNNWorkspaceExecutor end, ret: " << ret
                                    << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                                    << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
        return ret;
    } else {
        int ret = aclnnInplaceScatterGetWorkspaceSize(
            aclnnVariantPack.aclInTensors[0]->tensor, // self
            param_.dim, // scatter dim
            aclnnVariantPack.aclInTensors[1]->tensor, // index
            aclnnVariantPack.aclInTensors[2]->tensor, // src
            reduceType, // reduce type
            &this->aclnnOpCache_->workspaceSize,
            &this->aclnnOpCache_->aclExecutor);
        if (ret != atb::NO_ERROR) {
            ATB_SPEED_LOG_ERROR(opName_ << "aclnnInplaceScatterGetWorkspaceSize failed with error code: " << ret);
            return ret;
        }
        ATB_SPEED_LOG_DEBUG(opName_ << " aclnnScatterInplace SetAclNNWorkspaceExecutor end, ret: " << ret
                                    << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                                    << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
        return ret;
    }
}

int ScatterOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    if (!isInplace_) {
        int ret = aclnnScatter(
            workspace,
            this->aclnnOpCache_->workspaceSize,
            this->aclnnOpCache_->aclExecutor,
            stream);
        if (ret != 0) {
            ATB_SPEED_LOG_ERROR("aclnnScatter ExecuteAclNNOp failed, ret: " << ret);
        }
        return ret;
    } else {
        int ret = aclnnInplaceScatter(
            workspace,
            this->aclnnOpCache_->workspaceSize,
            this->aclnnOpCache_->aclExecutor,
            stream);
        if (ret != 0) {
            ATB_SPEED_LOG_ERROR("aclnnInplaceScatter ExecuteAclNNOp failed, ret: " << ret);
        }
        return ret;
    }
}

} // namespace common
} // namespace atb_speed
