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
#include "aclnnop/aclnn_dequant_rope_quant_kvcache.h"
#include "dequant_rope_quant_kvcache_operation.h"

namespace atb_speed {
namespace common {

DequantRopeQuantKvcacheOperation::DequantRopeQuantKvcacheOperation(
    const std::string &name,
    AclNNDequantRopeQuantKvcacheParam param) : AclNNOperation(name), param_(param) {}

DequantRopeQuantKvcacheOperation::~DequantRopeQuantKvcacheOperation()
{
    ATB_SPEED_LOG_DEBUG("DequantRopeQuantKvcacheOperation deconstructor");
    this->DestroyOperation();
}

atb::Status DequantRopeQuantKvcacheOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");

    for (int i = 0; i < NUM3; ++i) {
        outTensorDescs.at(i).format = inTensorDescs.at(0).format;
        outTensorDescs.at(i).dtype = inTensorDescs.at(1).dtype;
        outTensorDescs.at(i).shape.dimNum = NUM3;
    }

    const int64_t batchsize = inTensorDescs.at(DIM0).shape.dims[DIM0]; // x.size(0) [1024, 1280]
    const int64_t kvHeaddim = inTensorDescs.at(NUM4).shape.dims[DIM2]; // v_cache_ref.size(2); // 1
    const int64_t dim = inTensorDescs.at(NUM4).shape.dims[DIM3]; // v_cache_ref.size(3); // [9, 128, 1, 128]
    const int64_t qHeaddim = (dim == 0) ? 0 :
                                (inTensorDescs.at(DIM0).shape.dims[DIM1] - kvHeaddim * dim * NUM2) / dim;

    for (int i = 0; i < NUM3; ++i) {
        outTensorDescs.at(i).shape.dims[DIM0] = batchsize;
        outTensorDescs.at(i).shape.dims[DIM1] = (i > 0) ? kvHeaddim : qHeaddim;
        outTensorDescs.at(i).shape.dims[DIM2] = dim;
    }

    // translated aclInTensors translated
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        ATB_SPEED_LOG_DEBUG("Input tensor[" << i << "] address: "
                                           << aclnnVariantPack.aclInTensors.at(i)->tensor);
    }

    // translated aclOutTensors translated
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        ATB_SPEED_LOG_DEBUG("Output tensor[" << i << "] address: "
                                            << aclnnVariantPack.aclOutTensors.at(i)->tensor);
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return 0;
}

uint32_t DequantRopeQuantKvcacheOperation::GetInputNum() const
{
    return param_.enableDequant ? 12 : 10; // translateddequant: 12; translateddequant: 10
}

uint32_t DequantRopeQuantKvcacheOperation::GetOutputNum() const { return NUM3; }

int DequantRopeQuantKvcacheOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        if (i == 11) { // bias: 11
            aclnnTensor->tensorIdx++;
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

int DequantRopeQuantKvcacheOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        aclnnVariantPack.aclOutTensors[i] = CreateTensor(variantPack.outTensors.at(i), i);
        if (aclnnVariantPack.aclOutTensors[i]->tensor == nullptr) {
            return atb::ERROR_INTERNAL_ERROR;
        }
    }
    return atb::NO_ERROR;
}

int DequantRopeQuantKvcacheOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
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

int DequantRopeQuantKvcacheOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    char cacheMode[5] = "page";
    int ret = aclnnDequantRopeQuantKvcacheGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor,  // 0: x
        aclnnVariantPack.aclInTensors.at(1)->tensor,  // 1: cos
        aclnnVariantPack.aclInTensors.at(2)->tensor,  // 2: sin
        aclnnVariantPack.aclInTensors.at(3)->tensor,  // 3: k_cache
        aclnnVariantPack.aclInTensors.at(4)->tensor,  // 4: v_cache
        aclnnVariantPack.aclInTensors.at(5)->tensor,  // 5: indices
        aclnnVariantPack.aclInTensors.at(6)->tensor,  // 6: scale_k
        aclnnVariantPack.aclInTensors.at(7)->tensor,  // 7: scale_v
        aclnnVariantPack.aclInTensors.at(8)->tensor, // 8: offset_k
        aclnnVariantPack.aclInTensors.at(9)->tensor, // 9: offset_v
        param_.enableDequant ? aclnnVariantPack.aclInTensors.at(10)->tensor : nullptr,  // 10: weight_scale
        nullptr,  // 11: activation_scale
        param_.enableDequant ? aclnnVariantPack.aclInTensors.at(11)->tensor : nullptr,  // 12: bias
        aclCreateIntArray(param_.sizeSpilts.data(), param_.sizeSpilts.size()), // 13: sizeSpilts
        const_cast<char*>(param_.quantMode.c_str()), // 14: quantMode, char
        const_cast<char*>(param_.layout.c_str()),    // 15: layoutOptional
        param_.kvOutput, // 16: kvOutputOptional
        cacheMode, // 17: cachemode
        aclnnVariantPack.aclOutTensors.at(0)->tensor, // 18: qOut
        aclnnVariantPack.aclOutTensors.at(1)->tensor, // 19: kOut
        aclnnVariantPack.aclOutTensors.at(2)->tensor, // 20: vOut
        &this->aclnnOpCache_->workspaceSize, // 21: workspaceSize
        &this->aclnnOpCache_->aclExecutor); // 22: executor

    ATB_SPEED_LOG_DEBUG(opName_ << " end, ret:"
                               << ret << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                               << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int DequantRopeQuantKvcacheOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    int ret = aclnnDequantRopeQuantKvcache(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR("!!!!!!!!!!!! aclnnDequantRopeQuantKvcache failed, ret: " << ret);
    }
    return ret;
}
} // namespace common
} // namespace atb_speed
