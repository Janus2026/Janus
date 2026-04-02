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
#include "operations/aclnn/core/acl_nn_operation.h"
#include "aclnnop/aclnn_addmm.h"
#include "matmul_operation.h"

namespace atb_speed {
namespace common {

MatmulOperation::MatmulOperation(
    const std::string &name,
    AclNNMatmulParam param) : AclNNOperation(name), param_(param) {}

MatmulOperation::~MatmulOperation()
{
    ATB_SPEED_LOG_DEBUG("MatmulOperation deconstructor");
    this->DestroyOperation();
}

atb::Status MatmulOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                        atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "infer shape start");

    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    if (param_.outDataType == ACL_BF16 || inTensorDescs.at(DIM0).dtype == ACL_BF16) {
        outTensorDescs.at(DIM0).dtype = ACL_BF16;
    } else {
        outTensorDescs.at(DIM0).dtype = inTensorDescs.at(DIM0).dtype;
    }
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;

    int nDim = param_.transposeB ? DIM0 : DIM1; // inTensorDescs.at(DIM1).shape.dimNum translated 2
    if (inTensorDescs.at(0).shape.dimNum == DIM3) {
        ATB_SPEED_LOG_DEBUG("[input0 dimNum = 3] CHECK " << opName_ << " inputs shape: [input0]"
                       << inTensorDescs.at(DIM0).shape.dims[DIM0] << ", "
                       << inTensorDescs.at(DIM0).shape.dims[DIM1] << ", " << inTensorDescs.at(DIM0).shape.dims[DIM2]);
        ATB_SPEED_LOG_DEBUG("[input0 dimNum = 3] CHECK " << opName_ << " inputs shape: [input1]"
                       << inTensorDescs.at(DIM1).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM1).shape.dims[DIM1]);
        outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];
        outTensorDescs.at(DIM0).shape.dims[DIM2] = inTensorDescs.at(DIM1).shape.dims[nDim];
    } else if (inTensorDescs.at(0).shape.dimNum == DIM2) {
        ATB_SPEED_LOG_DEBUG("[input0 dimNum = 2] CHECK " << opName_ << " inputs shape: [input0]"
                       << inTensorDescs.at(DIM0).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM0).shape.dims[DIM1]);
        ATB_SPEED_LOG_DEBUG("[input0 dimNum = 2] CHECK " << opName_ << " inputs shape: [input1]"
                       << inTensorDescs.at(DIM1).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM1).shape.dims[DIM1]);
        outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM1).shape.dims[nDim];
    } else {
        ATB_SPEED_LOG_ERROR(opName_ << " invalid dim num:" << inTensorDescs.at(DIM0).shape.dimNum);
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return 0;
}

uint32_t MatmulOperation::GetInputNum() const
{
    uint32_t inputNum = DIM2;
    if (param_.hasBias) {
        inputNum += DIM1;
    }
    return inputNum;
}

uint32_t MatmulOperation::GetOutputNum() const
{
    return DIM1;
}

int MatmulOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
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

atb::Dims MatmulOperation::GetWeightStorageShape(const atb::TensorDesc atbTensorDesc) const
{
    atb::Dims storageTensorDims = atbTensorDesc.shape;  // NDtranslated,storageShapetranslatedoriginalShapetranslated
    if (atbTensorDesc.format == ACL_FORMAT_FRACTAL_NZ) {
        // nztranslated (k, n) => (n / 16, k / 16, 16, 16)
        // nztranslated (n, k) => (k / 16, n / 16, 16, 16)
        storageTensorDims.dimNum = NUM4;  // 4translated
        auto dim0 = atbTensorDesc.shape.dims[DIM0];
        uint32_t blockSize = 16;
        storageTensorDims.dims[DIM0] = atbTensorDesc.shape.dims[DIM1] / blockSize;
        storageTensorDims.dims[DIM1] = dim0 / blockSize;
        storageTensorDims.dims[DIM2] = blockSize;
        storageTensorDims.dims[DIM3] = blockSize;
    }
    return storageTensorDims;
}

int MatmulOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); i++) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(variantPack.inTensors.at(i));

        // StorageShape
        atb::Dims storageTensorDims = GetWeightStorageShape(squeezedAtbTensor.desc);

        // ViewShape and Stride
        atb::Dims viewDims = squeezedAtbTensor.desc.shape;
        if (i == 1 && this->param_.transposeB) {
            aclnnTensor->strides = GetTransposeTensorStride(viewDims);
            viewDims.dims[DIM0] = squeezedAtbTensor.desc.shape.dims[DIM1];
            viewDims.dims[DIM1] = squeezedAtbTensor.desc.shape.dims[DIM0];
        } else {
            aclnnTensor->strides = GetCopyTensorStride(viewDims);
        }

        aclnnTensor->tensor = aclCreateTensor(
            viewDims.dims, viewDims.dimNum, squeezedAtbTensor.desc.dtype,
            aclnnTensor->strides.data(), 0, squeezedAtbTensor.desc.format,
            storageTensorDims.dims, storageTensorDims.dimNum, squeezedAtbTensor.deviceData);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int MatmulOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
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
    return atb::NO_ERROR;
}

int MatmulOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    float zeroValue = 0.0f;
    float oneValue = 1.0f;
    aclScalar* betaZero = aclCreateScalar(&zeroValue, aclDataType::ACL_FLOAT);
    aclScalar* betaOne = aclCreateScalar(&oneValue, aclDataType::ACL_FLOAT);

    int ret = aclnnAddmmGetWorkspaceSize(
        param_.hasBias ? aclnnVariantPack.aclInTensors.at(DIM2)->tensor
                        : aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclInTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclInTensors.at(DIM1)->tensor,
        param_.hasBias ? betaOne : betaZero, betaOne,
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor, 0,
        &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);

    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int MatmulOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddmm start");
    int ret = aclnnAddmm(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddmm end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed
