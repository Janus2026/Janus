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
#include "aclnnop/aclnn_quant_matmul_v4.h"
#include "w8a8_operation.h"

namespace atb_speed {
namespace common {

W8A8Operation::W8A8Operation(
    const std::string &name,
    AclNNQuantMatmulParam param) : AclNNOperation(name), param_(param) {}

W8A8Operation::~W8A8Operation()
{
    ATB_SPEED_LOG_DEBUG("W8A8Operation deconstructor");
    this->DestroyOperation();
}

atb::Status W8A8Operation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                      atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    // translatedDequanttranslated, MMtranslatedINT_32
    outTensorDescs.at(0).dtype = param_.isOutDequantBias ? ACL_INT32 : param_.isBF16 ? ACL_BF16 : ACL_FLOAT16;

    int nDim = param_.transposeB ? DIM0 : DIM1;
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

uint32_t W8A8Operation::GetInputNum() const
{
    uint32_t inputNum = 3;
    ATB_SPEED_LOG_DEBUG("initial inputNum: " << inputNum);
    if (param_.hasPerTokenScale) {
        ATB_SPEED_LOG_DEBUG("QuantBatchMatmul & hasPerTokenScale");
        ++inputNum;
    }
    if (param_.hasBias) {
        ATB_SPEED_LOG_DEBUG("QuantBatchMatmul & hasBias");
        ++inputNum;
    }
    ATB_SPEED_LOG_DEBUG("final inputNum: " << inputNum);
    return inputNum;
}

uint32_t W8A8Operation::GetOutputNum() const { return NUM1; }

int W8A8Operation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
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

atb::Dims W8A8Operation::GetWeightStorageShape(const atb::TensorDesc atbTensorDesc) const
{
    atb::Dims storageTensorDims = atbTensorDesc.shape;  // NDtranslated,storageShapetranslatedoriginalShapetranslated
    // NDtranslatedNZ
    if (atbTensorDesc.format == ACL_FORMAT_FRACTAL_NZ) {
        // nztranslated (k, n) => (n / 32, k / 16, 16, 32)
        // nztranslated (n, k) => (k / 32, n / 16, 16, 32)
        storageTensorDims.dimNum = NUM4;  // 4translated
        auto dim0 = atbTensorDesc.shape.dims[DIM0];
        // m0,n0translated:float16:n0=m0=16, int8:n0=32,m0=16
        uint32_t blockSize = 16;   // m0, translated
        uint32_t n0 = 32;          // n0, translated, w8a8translatedint8
        storageTensorDims.dims[DIM0] = atbTensorDesc.shape.dims[DIM1] / n0;
        storageTensorDims.dims[DIM1] = dim0 / blockSize;
        storageTensorDims.dims[DIM2] = blockSize;
        storageTensorDims.dims[DIM3] = n0;
    }
    return storageTensorDims;
}

int W8A8Operation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor atbTensor = variantPack.inTensors.at(i);

        if (param_.matmulBackend == atb_speed::common::OpBackend::ACLNN) {
            // StorageShape
            atb::Dims storageTensorDims = GetWeightStorageShape(atbTensor.desc);

            // ViewShape and Stride
            atb::Dims viewDims = atbTensor.desc.shape;
            // aclInTensors[1]translatedweight
            if (i == 1 && this->param_.transposeB) {
                aclnnTensor->strides = GetTransposeTensorStride(viewDims);
                viewDims.dims[DIM0] = atbTensor.desc.shape.dims[DIM1];
                viewDims.dims[DIM1] = atbTensor.desc.shape.dims[DIM0];
            } else {
                aclnnTensor->strides = GetCopyTensorStride(viewDims);
            }
            // offsettranslated0
            aclnnTensor->tensor = aclCreateTensor(
                viewDims.dims, viewDims.dimNum, atbTensor.desc.dtype,
                aclnnTensor->strides.data(), 0, atbTensor.desc.format,
                storageTensorDims.dims, storageTensorDims.dimNum, atbTensor.deviceData);
        } else {
            aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
            aclnnTensor->tensor = aclCreateTensor(
                atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.desc.dtype,
                aclnnTensor->strides.data(), 0, atbTensor.desc.format, atbTensor.desc.shape.dims,
                atbTensor.desc.shape.dimNum, atbTensor.deviceData);
        }

        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int W8A8Operation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(NUM1);
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

int W8A8Operation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    uint32_t inputIdx = 3;
    aclTensor* perTokenScaleTensor = param_.hasPerTokenScale ? \
        aclnnVariantPack.aclInTensors.at(inputIdx++)->tensor : nullptr;
    // translatedDequanttranslated,biasTensortranslatednullptr
    aclTensor* biasTensor = param_.isOutDequantBias ? nullptr : \
                            param_.hasBias ? aclnnVariantPack.aclInTensors.at(inputIdx++)->tensor : nullptr;
    int ret = aclnnQuantMatmulV4GetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor,  // 0: input
        aclnnVariantPack.aclInTensors.at(1)->tensor,  // 1: weight
        aclnnVariantPack.aclInTensors.at(2)->tensor,  // 2: scale
        nullptr,  // offset
        perTokenScaleTensor,  // per token scale
        biasTensor,  // bias
        false, // transposeX1
        param_.transposeB, // transposeX2
        aclnnVariantPack.aclOutTensors.at(0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " end, ret:"
                  << ret << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int W8A8Operation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    int ret = aclnnQuantMatmulV4(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR("ExecuteAclNNOp failed, ret: " << ret);
    }
    return ret;
}

} // namespace common
} // namespace atb_speed