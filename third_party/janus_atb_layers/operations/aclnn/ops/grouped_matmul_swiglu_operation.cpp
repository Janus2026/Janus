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
#include "grouped_matmul_swiglu_operation.h"
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {

GroupedMatmulSwigluOperation::GroupedMatmulSwigluOperation(
    const std::string &name,
    AclNNGroupedSwigluMatmulParam param) : AclNNOperation(name), param_(param) {
}

GroupedMatmulSwigluOperation::~GroupedMatmulSwigluOperation() {
}

atb::Status GroupedMatmulSwigluOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "GroupedMatmulSwigluOperation infer shape start");
    ATB_SPEED_LOG_DEBUG(opName_ << "exports" << inTensorDescs.at(DIM2).shape.dims[DIM0]);
    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = inTensorDescs.at(DIM0).dtype;
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;
    outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
    int nDim = DIM1;
    if (inTensorDescs.at(DIM1).shape.dims[1] == inTensorDescs.at(DIM0).shape.dims[1]) {
        nDim = DIM2;
    }
    outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM1).shape.dims[nDim] / 2; // 2: swiglu quant
    outTensorDescs.at(DIM1).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM1).dtype = ACL_FLOAT;
    outTensorDescs.at(DIM1).shape.dimNum = 1;
    outTensorDescs.at(DIM1).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];

    ATB_SPEED_LOG_DEBUG(opName_ << "GroupedMatmulSwigluOperation infer shape end");
    return 0;
}

uint32_t GroupedMatmulSwigluOperation::GetInputNum() const
{
    return INPUT_NUM;
}

uint32_t GroupedMatmulSwigluOperation::GetOutputNum() const
{
    return OUTPUT_NUM;
}

atb::Dims SetWeightStorageShape(const atb::TensorDesc& atbTensorDesc)
{
    atb::Dims storageTensorDims = atbTensorDesc.shape;  // NDtranslated,storageShapetranslatedoriginalShapetranslated
    if (atbTensorDesc.format == ACL_FORMAT_FRACTAL_NZ) {
        // nztranslated
        storageTensorDims.dimNum = 5;  // 5: 5translated
        // (group_size, k, n) => (group_size, n / 16, k / 32, 16, 32)
        storageTensorDims.dims[0] = atbTensorDesc.shape.dims[0];
        storageTensorDims.dims[2] = 1 + ((atbTensorDesc.shape.dims[1] - 1) / 16);  // 1, 16:2: translated, 16: paddingtranslated
        storageTensorDims.dims[1] = 1 + ((atbTensorDesc.shape.dims[2] - 1) / 32);  // 2, 32:2: translated, 32: paddingtranslated
        storageTensorDims.dims[3] = 16;  // 3, 16:NZtranslated
        storageTensorDims.dims[4] = 32;  // 4, 32:NZtranslated
    }
    return storageTensorDims;
}

atb::Status GroupedMatmulSwigluOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(INPUT_NUM);
    const int aclnnTensorIndex[INPUT_NUM] = {0, 1, 4, 5, 6}; // valid input index
    for (size_t i = 0; i < INPUT_NUM; i++) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = aclnnTensorIndex[i];
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor squeezedAtbTensor = variantPack.inTensors.at(i);

        // StorageShape
        atb::Dims storageTensorDims = SetWeightStorageShape(squeezedAtbTensor.desc);

        // ViewShape and Stride
        atb::Dims viewDims = squeezedAtbTensor.desc.shape;
        if (squeezedAtbTensor.desc.shape.dimNum >= 3 && this->param_.transposeB) {  // 3: translated
            aclnnTensor->strides = GetTransposeTensorStride(viewDims);
            viewDims.dims[0] = squeezedAtbTensor.desc.shape.dims[0];
            viewDims.dims[1] = squeezedAtbTensor.desc.shape.dims[2];  // 1, 2: translated
            viewDims.dims[2] = squeezedAtbTensor.desc.shape.dims[1];  // 1, 2: translated
        } else {
            aclnnTensor->strides = GetCopyTensorStride(viewDims);
        }

        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(storageTensorDims, storageTensorDims,
            squeezedAtbTensor, aclnnTensor));
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

atb::Status GroupedMatmulSwigluOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(OUTPUT_NUM);
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        atb::Tensor squeezedAtbTensor = variantPack.outTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(squeezedAtbTensor.desc.shape, squeezedAtbTensor.desc.shape,
            squeezedAtbTensor, aclnnTensor));
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int GroupedMatmulSwigluOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnGroupedMatmulSwigluQuantGetWorkspaceSize(aclnnVariantPack.aclInTensors.at(DIM0)->tensor,  // 0: x
        aclnnVariantPack.aclInTensors.at(DIM1)->tensor,  // 1: weight
        nullptr,  // bias
        nullptr,  // offset
        aclnnVariantPack.aclInTensors.at(DIM2)->tensor,  // 2: weight_scale
        aclnnVariantPack.aclInTensors.at(DIM3)->tensor,  // 3: x_scale
        aclnnVariantPack.aclInTensors.at(4)->tensor, // 4: group_list IDX
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,  // out0: output
        aclnnVariantPack.aclOutTensors.at(DIM1)->tensor,  // out1: output_scale
        nullptr,  // out2: output_offset
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);

    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int GroupedMatmulSwigluOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnGroupedMatmul start");
    int ret = aclnnGroupedMatmulSwigluQuant(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnGroupedMatmul end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed
