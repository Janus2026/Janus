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
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "aclnnop/aclnn_dequant_swiglu_quant.h"
#include "dequant_swiglu_quant_operation.h"

namespace atb_speed {
namespace common {

DequantSwigluQuantOperation::DequantSwigluQuantOperation(
    const std::string &name,
    AclNNDequantSwigluQuantParam param) : AclNNOperation(name), param_(param) {}

DequantSwigluQuantOperation::~DequantSwigluQuantOperation()
{
    ATB_SPEED_LOG_DEBUG("DequantSwigluQuantOperation deconstructor");
    this->DestroyOperation();
}

atb::Status DequantSwigluQuantOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = aclDataType::ACL_INT8;
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;
    outTensorDescs.at(DIM1).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM1).dtype = aclDataType::ACL_FLOAT;
    outTensorDescs.at(DIM1).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;
    if (inTensorDescs.at(DIM0).shape.dimNum == DIM3) {
        outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];
        outTensorDescs.at(DIM0).shape.dims[DIM2] = inTensorDescs.at(DIM0).shape.dims[DIM2] / NUM2;
        outTensorDescs.at(DIM1).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(DIM1).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];
        outTensorDescs.at(DIM1).shape.dims[DIM2] = NUM1;
    } else if (inTensorDescs.at(DIM0).shape.dimNum == DIM2) {
        outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1] / NUM2;
        outTensorDescs.at(DIM1).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(DIM1).shape.dims[DIM1] = NUM1;
    } else {
        ATB_SPEED_LOG_ERROR(opName_ << " invalid dim num:" << inTensorDescs.at(DIM0).shape.dimNum);
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return 0;
}


uint32_t DequantSwigluQuantOperation::GetInputNum() const
{
    if (param_.inTensorsNum < NUM1 || param_.inTensorsNum > NUM5) {
        ATB_SPEED_LOG_DEBUG("DequantSwigluQuantOperation param inTensorsNum is wrong! reset to 5.");
        return NUM5;
    }
    return param_.inTensorsNum;
}

uint32_t DequantSwigluQuantOperation::GetOutputNum() const { return NUM2; }

int DequantSwigluQuantOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        if (param_.inTensorsNum == NUM3 && i > 0) {
            aclnnTensor->tensorIdx = i + NUM3;
        } else if (param_.inTensorsNum == NUM5 && i > 1) {
            aclnnTensor->tensorIdx = i + 1;
        } else {
            aclnnTensor->tensorIdx = i;
        }
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor atbTensor = variantPack.inTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
        aclnnTensor->tensor = aclCreateTensor(
            atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.desc.dtype,
            aclnnTensor->strides.data(), 0, atbTensor.desc.format, atbTensor.desc.shape.dims,
            atbTensor.desc.shape.dimNum, atbTensor.deviceData);

        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int DequantSwigluQuantOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        atb::Tensor atbTensor = variantPack.outTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
        aclnnTensor->tensor = aclCreateTensor(
            atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.desc.dtype,
            aclnnTensor->strides.data(), 0, atbTensor.desc.format, atbTensor.desc.shape.dims,
            atbTensor.desc.shape.dimNum, atbTensor.deviceData);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " OutTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int DequantSwigluQuantOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
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

int DequantSwigluQuantOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    uint32_t inputIdx = 0;
    aclTensor* xTensor = aclnnVariantPack.aclInTensors.at(inputIdx++)->tensor;
    aclTensor* weightScaleTensor = (param_.inTensorsNum > NUM3) ?
        aclnnVariantPack.aclInTensors.at(inputIdx++)->tensor : nullptr;
    aclTensor* biasTensor = (param_.inTensorsNum > NUM3) ?
        aclnnVariantPack.aclInTensors.at(inputIdx++)->tensor : nullptr;
    aclTensor* quantScaleTensor = (param_.inTensorsNum > NUM1) ?
        aclnnVariantPack.aclInTensors.at(inputIdx++)->tensor : nullptr;
    aclTensor* quantOffsetTensor = (param_.inTensorsNum > NUM1) ?
        aclnnVariantPack.aclInTensors.at(inputIdx++)->tensor : nullptr;

    int ret = aclnnDequantSwigluQuantGetWorkspaceSize(
        xTensor,  // x
        weightScaleTensor,  // weightScaleOptional
        nullptr,  // activationScaleOptional
        biasTensor,  // biasOptional
        quantScaleTensor,  // quantScaleOptional
        quantOffsetTensor,  // quantOffsetOptional
        nullptr,  // groupIndexOptional
        param_.activateLeft,  // activateLeft
        const_cast<char*>(param_.quantMode.c_str()),  // quantMode, char
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,  // y
        aclnnVariantPack.aclOutTensors.at(DIM1)->tensor,  // scaleOptional
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " end, ret:"
                  << ret << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int DequantSwigluQuantOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " DequantSwigluQuantOperation start");
    int ret = aclnnDequantSwigluQuant(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " DequantSwigluQuantOperation end, ret: " << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed