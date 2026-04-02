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
#include "moe_grouped_matmul_weight_nz_operation.h"
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "aclnn_moe_grouped_matmul_weight_nz.h"
#include "operations/aclnn/utils/utils.h"
#include "atb_speed/utils/check_util.h"
#include "moe_grouped_matmul_weight_nz_operation.h"

namespace atb_speed {
namespace common {

MoeGroupedMatmulWeightNZOperation::MoeGroupedMatmulWeightNZOperation(
    const std::string &name,
    bool transposeB) : AclNNOperation(name) {
        this->transposeB = transposeB;
}

MoeGroupedMatmulWeightNZOperation::~MoeGroupedMatmulWeightNZOperation() {
}

atb::Status MoeGroupedMatmulWeightNZOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "MoeGroupedMatmulWeightNZOperation infer shape start");
    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = ACL_BF16;
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;
    int nDim = this->transposeB ? DIM1 : DIM2;
    outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
    outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM1).shape.dims[nDim];

    ATB_SPEED_LOG_DEBUG(opName_ << "MoeGroupedMatmulWeightNZOperation infer shape end");
    return 0;
}

uint32_t MoeGroupedMatmulWeightNZOperation::GetInputNum() const
{
    return NUM3;
}

uint32_t MoeGroupedMatmulWeightNZOperation::GetOutputNum() const
{
    return NUM1;
}

atb::Dims MoeGroupedMatmulWeightNZOperation::GetWeightStorageShape(const atb::TensorDesc atbTensorDesc,bool is_bf16)
{
    atb::Dims storageTensorDims = atbTensorDesc.shape;  // NDtranslated,storageShapetranslatedoriginalShapetranslated
    if (atbTensorDesc.format == ACL_FORMAT_FRACTAL_NZ) {
        // nztranslated
        storageTensorDims.dimNum = 5;  // 5: 5translated
        // (group_size, k, n) => (group_size, k / 16, n / 16, 16, 16)
        // (group_size, n, k) => (group_size, n / 16, k / 16, 16, 16)
        storageTensorDims.dims[0] = atbTensorDesc.shape.dims[0];
        storageTensorDims.dims[1] = 1 + ((atbTensorDesc.shape.dims[1] - 1) / 16);  // 1, 16:1: translated, 16: paddingtranslated
        storageTensorDims.dims[3] = 16;  // 3, 16:NZtranslated
        if (is_bf16) {
          storageTensorDims.dims[2] = 1 + ((atbTensorDesc.shape.dims[2] - 1) / 32);  // 2, 16:1: translated, 16: paddingtranslated
          storageTensorDims.dims[4] = 32;  // 4, 16:NZtranslated
        } else {
          storageTensorDims.dims[2] = 1 + ((atbTensorDesc.shape.dims[2] - 1) / 16);  // 2, 16:1: translated, 16: paddingtranslated
          storageTensorDims.dims[4] = 16;  // 4, 16:NZtranslated
        }
    }
    return storageTensorDims;
}

atb::Status MoeGroupedMatmulWeightNZOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    inputVectorOfTensor.resize(GetInputNum());
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    const size_t inTensorCount = aclnnVariantPack.aclInTensors.size();

    for (size_t i = 0; i < inTensorCount; i++) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        if (i == inTensorCount - 1) {
            aclnnTensor->tensorIdx = 7;
        } else {
            aclnnTensor->tensorListidx = i;
            aclnnTensor->tensorIdx = 0;
        }
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor squeezedAtbTensor = variantPack.inTensors.at(i);

        // StorageShape
        atb::Dims storageTensorDims;
        storageTensorDims = GetWeightStorageShape(squeezedAtbTensor.desc, true);
        atb::Dims viewDims = squeezedAtbTensor.desc.shape;
        aclnnTensor->strides = GetCopyTensorStride(viewDims);

        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(viewDims, storageTensorDims, squeezedAtbTensor, aclnnTensor));
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    
    aclnnVariantPack.aclInTensorList.clear();
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size() - 1; i++) {
        inputVectorOfTensor.at(i).clear();
        inputVectorOfTensor.at(i).push_back(aclnnVariantPack.aclInTensors.at(i)->tensor);
        aclnnVariantPack.aclInTensorList.push_back(aclCreateTensorList(
            inputVectorOfTensor.at(i).data(), inputVectorOfTensor.at(i).size()));
    }

    return atb::NO_ERROR;
}

atb::Status MoeGroupedMatmulWeightNZOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    const size_t outTensorCount = aclnnVariantPack.aclOutTensors.size();
    for (size_t i = 0; i < outTensorCount; i++) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorListidx = i;
        aclnnTensor->tensorIdx = 0;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        atb::Tensor squeezedAtbTensor = variantPack.outTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);

        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(squeezedAtbTensor.desc.shape, squeezedAtbTensor.desc.shape,
            squeezedAtbTensor, aclnnTensor));
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    yTensorVector.clear();
    yTensorVector.push_back(aclnnVariantPack.aclOutTensors.at(DIM0)->tensor);
    aclnnVariantPack.aclOutTensorList.clear();
    aclnnVariantPack.aclOutTensorList.push_back(aclCreateTensorList(yTensorVector.data(), yTensorVector.size()));
    return atb::NO_ERROR;
}

int MoeGroupedMatmulWeightNZOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = 0;

    ret = aclnnMoeGroupedMatmulWeightNzGetWorkspaceSize(
        aclnnVariantPack.aclInTensorList.at(DIM0),
        aclnnVariantPack.aclInTensorList.at(DIM1),
        aclnnVariantPack.aclInTensors.at(DIM2)->tensor,
        this->transposeB,
        aclnnVariantPack.aclOutTensorList.at(DIM0),
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);

    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int MoeGroupedMatmulWeightNZOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnGroupedMatmul start");
    int ret = 0;

    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    ret = aclnnMoeGroupedMatmulWeightNz(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);

    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnGroupedMatmul end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed
