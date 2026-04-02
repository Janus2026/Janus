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
#include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"
#include "quant_batch_matmul_operation.h"

namespace atb_speed {
namespace common {

QuantBatchMatmulOperation::QuantBatchMatmulOperation(
    const std::string &name,
    AclNNWeightQuantBatchMatmulParam param) : AclNNOperation(name), param_(param) {}

QuantBatchMatmulOperation::~QuantBatchMatmulOperation()
{
    ATB_SPEED_LOG_DEBUG("QuantBatchMatmulOperation deconstructor");
    this->DestroyOperation();
}

atb::Status QuantBatchMatmulOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;

    int nDim = param_.transposeB ? DIM0 : DIM1;
    if (inTensorDescs.at(0).shape.dimNum == DIM3) {
        ATB_SPEED_LOG_DEBUG("[input0 dimNum = 3] CHECK " << opName_ << " inputs shape: [input0]"
                       << inTensorDescs.at(DIM0).shape.dims[DIM0] << ", "
                       << inTensorDescs.at(DIM0).shape.dims[DIM1] << ", " << inTensorDescs.at(DIM0).shape.dims[DIM2]);
        ATB_SPEED_LOG_DEBUG("[input0 dimNum = 3] CHECK " << opName_ << " inputs shape: [input1]"
                       << inTensorDescs.at(DIM1).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM1).shape.dims[DIM1]);
        outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];
        outTensorDescs.at(DIM0).shape.dims[DIM2] = inTensorDescs.at(DIM3).shape.dims[nDim];
    } else if (inTensorDescs.at(0).shape.dimNum == DIM2) {
        ATB_SPEED_LOG_DEBUG("[input0 dimNum = 2] CHECK " << opName_ << " inputs shape: [input0]"
                       << inTensorDescs.at(DIM0).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM0).shape.dims[DIM1]);
        ATB_SPEED_LOG_DEBUG("[input0 dimNum = 2] CHECK " << opName_ << " inputs shape: [input1]"
                       << inTensorDescs.at(DIM1).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM1).shape.dims[DIM1]);
        outTensorDescs.at(0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(0).shape.dims[DIM1] = inTensorDescs.at(DIM3).shape.dims[nDim];
    } else {
        ATB_SPEED_LOG_ERROR(opName_ << " invalid dim num:" << inTensorDescs.at(DIM0).shape.dimNum);
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return 0;
}

uint32_t QuantBatchMatmulOperation::GetInputNum() const { return param_.hasBias ? NUM5 : NUM4; }

uint32_t QuantBatchMatmulOperation::GetOutputNum() const { return NUM1; }

atb::Dims QuantBatchMatmulOperation::GetWeightStorageShape(const atb::TensorDesc atbTensorDesc)
{
    if (atbTensorDesc.format != ACL_FORMAT_FRACTAL_NZ) {
        // ndtranslated,storageShapetranslatedoriginalShapetranslated
        return atbTensorDesc.shape;
    }
    // nztranslated
    atb::Dims storageTensorDims = atbTensorDesc.shape;
    storageTensorDims.dimNum = 4;  // 4: 4translated
    if (param_.transposeB) {
        uint32_t kPadding = 16;
        uint32_t nPadding = 32;
        // (n, k) => (k1, n1, n0, k0)
        storageTensorDims.dims[0] = 1 + ((atbTensorDesc.shape.dims[1] - 1) / kPadding);
        storageTensorDims.dims[1] = 1 + ((atbTensorDesc.shape.dims[0] - 1) / nPadding);
        storageTensorDims.dims[2] = nPadding;  // 2: translated
        storageTensorDims.dims[3] = kPadding;  // 3: translated
    } else {
        uint32_t kPadding = 32;
        uint32_t nPadding = 16;
        // (k, n) => (n1, k1, k0, n0)
        storageTensorDims.dims[0] = 1 + ((atbTensorDesc.shape.dims[1] - 1) / nPadding);
        storageTensorDims.dims[1] = 1 + ((atbTensorDesc.shape.dims[0] - 1) / kPadding);
        storageTensorDims.dims[2] = kPadding;  // 2: translated
        storageTensorDims.dims[3] = nPadding;  // 3: translated
    }
    return storageTensorDims;
}

atb::Status QuantBatchMatmulOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = (i == 4) ? (i + 2) : i;  // 4, 2: biastranslatedaclExecutortranslatedidxtranslated6
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor preprocessedATBTensor = this->PreprocessATBInTensor(variantPack.inTensors.at(i), i);
        if ((i == 1) || (i == 2) || (i == 3)) {  // 1, 2, 3: weight, weight_scale, weight_offset
            if (preprocessedATBTensor.desc.shape.dimNum != NUM2) {
                ATB_SPEED_LOG_ERROR(this->opName_ << " weight tensor dimNum after combine batch size "
                               << "and seq len axis should be 2, but got " << preprocessedATBTensor.desc.shape.dimNum);
                return atb::ERROR_INTERNAL_ERROR;
            }
            // StorageShape
            atb::Dims storageDims = preprocessedATBTensor.desc.shape;
            if (i == 1) {  // weighttranslatedstorageShapetranslatedNZtranslatedNDtranslated
                storageDims = GetWeightStorageShape(preprocessedATBTensor.desc);
            }
            // ViewShape and Stride
            atb::Dims viewDims = preprocessedATBTensor.desc.shape;
            if (IsA2() && this->param_.transposeB) {
                aclnnTensor->strides = GetTransposeTensorStride(viewDims);
                viewDims.dims[0] = preprocessedATBTensor.desc.shape.dims[1];
                viewDims.dims[1] = preprocessedATBTensor.desc.shape.dims[0];
            } else {
                aclnnTensor->strides = GetCopyTensorStride(viewDims);
            }
            CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(
                viewDims, storageDims, preprocessedATBTensor, aclnnTensor));
        } else {
            aclnnTensor->strides = GetCopyTensorStride(preprocessedATBTensor.desc.shape);
            CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(preprocessedATBTensor.desc.shape,
                preprocessedATBTensor.desc.shape, preprocessedATBTensor, aclnnTensor));
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

atb::Status QuantBatchMatmulOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(NUM1);
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(variantPack.outTensors.at(i));
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(squeezedAtbTensor.desc.shape,
            squeezedAtbTensor.desc.shape, squeezedAtbTensor, aclnnTensor));
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int QuantBatchMatmulOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor,  // 0: x
        aclnnVariantPack.aclInTensors.at(1)->tensor,  // 1: weight
        aclnnVariantPack.aclInTensors.at(2)->tensor,  // 2: antiquantScale
        aclnnVariantPack.aclInTensors.at(3)->tensor, nullptr, nullptr,  // 3: antiquantOffset
        param_.hasBias ? aclnnVariantPack.aclInTensors.at(4)->tensor : nullptr,  // 4: bias
        param_.quantGroupSize, aclnnVariantPack.aclOutTensors.at(0)->tensor,
        &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " end, ret:"
                  << ret << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int QuantBatchMatmulOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    return aclnnWeightQuantBatchMatmulV2(
        workspace, this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor, stream);
}
} // namespace common
} // namespace atb_speed