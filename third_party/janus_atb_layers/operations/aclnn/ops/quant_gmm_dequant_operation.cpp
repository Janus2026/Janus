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
#include <unistd.h>
#include <cmath>

#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_grouped_matmul_dequant.h"
#include "quant_gmm_dequant_operation.h"

namespace atb_speed {
namespace common {


QuantGMMDequantOperation::QuantGMMDequantOperation(
    const std::string &name,
    AclNNQuantGMMDequantParam param) : AclNNOperation(name), param_(param) {}


QuantGMMDequantOperation::~QuantGMMDequantOperation()
{
    ATB_SPEED_LOG_DEBUG("QuantGMMDequantOperation deconstructor");
    this->DestroyOperation();
}

atb::Status QuantGMMDequantOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "QuantGMMDequantOperation infer shape start");
    outTensorDescs.at(0).format = inTensorDescs.at(0).format; // FORMAT_ND
    outTensorDescs.at(0).dtype = param_.outDataType; // ACL_FLOAT16;
    // in1 (8192, 7168); in2 [64, 2048, 7168]; out0 (8192, 2048)
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum; // dimNum = 2
    outTensorDescs.at(0).shape.dims[DIM0] = inTensorDescs.at(0).shape.dims[DIM0]; // DIM0 = inTensor0.DIM0
    outTensorDescs.at(0).shape.dims[DIM1] = inTensorDescs.at(NUM1).shape.dims[param_.transposeB ? DIM1 : DIM2];

    for (uint64_t i = 0; i < GetInputNum(); ++i) {
        ATB_SPEED_LOG_DEBUG(opName_ << " QuantGMMDequantOperation infer shape end" <<
                            " format: " << inTensorDescs.at(i).format <<
                            " dtype: " << inTensorDescs.at(i).dtype <<
                            " dimNum: " << inTensorDescs.at(i).shape.dimNum <<
                            " dim0: " << inTensorDescs.at(i).shape.dims[0] <<
                            " dim1: " << inTensorDescs.at(i).shape.dims[1]
                            );
        if (i == 1) {
            ATB_SPEED_LOG_DEBUG(" dim2: " << inTensorDescs.at(1).shape.dims[2]);
        }
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " QuantGMMDequantOperation infer shape end" <<
                        " format: " << outTensorDescs.at(0).format <<
                        " dtype: " << outTensorDescs.at(0).dtype <<
                        " dimNum: " << outTensorDescs.at(0).shape.dimNum <<
                        " dims: " << outTensorDescs.at(0).shape.dims[0] <<
                        " dims: " << outTensorDescs.at(0).shape.dims[1]);
    return 0;
}

uint32_t QuantGMMDequantOperation::GetInputNum() const
{
    return NUM4;
}

uint32_t QuantGMMDequantOperation::GetOutputNum() const
{
    return NUM1;
}

atb::Dims QuantGMMDequantOperation::GetWeightStorageShape(const atb::TensorDesc atbTensorDesc) const
{
    atb::Dims storageTensorDims = atbTensorDesc.shape;  // NDtranslated,storageShapetranslatedoriginalShapetranslated
    ATB_SPEED_LOG_DEBUG(opName_ << " GetWeightStorageShape inWeightTensor dim: " <<
        atbTensorDesc.shape.dims[0] << ", " << atbTensorDesc.shape.dims[1] << ", " << atbTensorDesc.shape.dims[2]
    );

    if (atbTensorDesc.format == ACL_FORMAT_FRACTAL_NZ) {
        // nztranslated
        storageTensorDims.dimNum = 5;  // 5translated
        // (group_size, n, k) => (group_size, k / 32, n / 16, 16, 32)
        storageTensorDims.dims[0] = atbTensorDesc.shape.dims[0];
        storageTensorDims.dims[3] = 16;  // 3, 16:NZtranslated
        storageTensorDims.dims[4] = 32;  // 4, 16:NZtranslated
        if (param_.transposeB) {
            storageTensorDims.dims[1] = ((atbTensorDesc.shape.dims[2] + 32 - 1) / 32);  // 1, 32:1: translated, 32: paddingtranslated
            storageTensorDims.dims[2] = ((atbTensorDesc.shape.dims[1] + 16 - 1) / 16);  // 2, 16:1: translated, 16: paddingtranslated
        } else {
            storageTensorDims.dims[1] = ((atbTensorDesc.shape.dims[1] + 32 - 1) / 32);  // 1, 32:1: translated, 32: paddingtranslated
            storageTensorDims.dims[2] = ((atbTensorDesc.shape.dims[2] + 16 - 1) / 16);  // 2, 16:1: translated, 16: paddingtranslated
        }
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " GetWeightStorageShape storageTensorDims dims: " <<
        storageTensorDims.dims[0] << ", " << storageTensorDims.dims[1] << ", " << storageTensorDims.dims[2] << ", " <<
        storageTensorDims.dims[3] << ", " << storageTensorDims.dims[4]
    );
    return storageTensorDims;
}

atb::Status QuantGMMDequantOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->tensorListidx = AclNNTensor::notInTensorList;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor atbTensor = variantPack.inTensors.at(i);
        // StorageShape
        if (i == 1) {
            atb::Tensor storageATBTensor =  variantPack.inTensors.at(i);
            atb::Dims storageTensorDims = GetWeightStorageShape(storageATBTensor.desc);
            aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
            CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(storageTensorDims, storageTensorDims,
                atbTensor, aclnnTensor)); // gmm translated viewDims.dimNum translated
        } else {
            aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
            CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape,
                atbTensor, aclnnTensor));
        }

        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

atb::Status QuantGMMDequantOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->tensorListidx = AclNNTensor::notInTensorList;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        atb::Tensor atbTensor = variantPack.outTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape,
            atbTensor, aclnnTensor));
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int QuantGMMDequantOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;

    int ret = aclnnQuantGroupedMatmulDequantGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor, // 0: x
        aclnnVariantPack.aclInTensors.at(NUM1)->tensor, // 1: weight
        aclnnVariantPack.aclInTensors.at(NUM2)->tensor, // 2: weightScale
        aclnnVariantPack.aclInTensors.at(NUM3)->tensor, // 3: groupList, int64
        nullptr, nullptr, nullptr, nullptr, // 4: bias; 5: xScale; 6: xOffset; 7: smoothScale;
        param_.quantMode.data(), // 8: xQuantMode
        param_.transposeB, // 9: transposeWeight
        aclnnVariantPack.aclOutTensors.at(0)->tensor, // 10: out
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);

    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}


int QuantGMMDequantOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " QuantGMMDequantOperation start");
    int ret = aclnnQuantGroupedMatmulDequant(workspace, this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " QuantGMMDequantOperation end, ret:" << ret);
    return 0;
}

} // namespace common
} // namespace atb_speed