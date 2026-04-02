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

#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "rms_norm_operation.h"

namespace atb_speed {
namespace common {

RmsNormOperation::RmsNormOperation(const std::string &name, float epsilon) : AclNNOperation(name)
{
    this->opName_ = name;
    this->epsilon = epsilon;
    
}

atb::Status RmsNormOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                            atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
    ATB_SPEED_LOG_DEBUG("inTensorDescs.at(0).shape.dimNum" << inTensorDescs.at(0).shape.dimNum);
    for (size_t i = 0; i < outTensorDescs.size(); i++) {
        outTensorDescs.at(i).format = inTensorDescs.at(0).format;
        if (i == NUM1) {
            outTensorDescs.at(i).dtype = aclDataType::ACL_FLOAT;
        } else {
            outTensorDescs.at(i).dtype = inTensorDescs.at(0).dtype;
        }

        outTensorDescs.at(i).shape.dimNum = inTensorDescs.at(0).shape.dimNum;

        if (inTensorDescs.at(0).shape.dimNum == DIM3) {
            ATB_SPEED_LOG_DEBUG("[input0 dimNum = 3] CHECK W8A16_OP inputs shape: [input0]"
                           << inTensorDescs.at(0).shape.dims[DIM0] << ", " << inTensorDescs.at(0).shape.dims[DIM1]
                           << ", " << inTensorDescs.at(0).shape.dims[DIM2]);
            outTensorDescs.at(i).shape.dims[DIM0] = inTensorDescs.at(0).shape.dims[DIM0];
            outTensorDescs.at(i).shape.dims[DIM1] = inTensorDescs.at(0).shape.dims[DIM1];
            outTensorDescs.at(i).shape.dims[DIM2] = inTensorDescs.at(0).shape.dims[DIM2];
            ATB_SPEED_LOG_DEBUG("[output0 dimNum = 3] CHECK W8A16_OP inputs shape: [output0]"
                           << outTensorDescs.at(0).shape.dims[DIM0] << ", " << outTensorDescs.at(0).shape.dims[DIM1]
                           << ", " << outTensorDescs.at(0).shape.dims[DIM2]);
            ATB_SPEED_LOG_DEBUG("[output1 dimNum = 3] CHECK W8A16_OP inputs shape: [output1]"
                           << outTensorDescs.at(1).shape.dims[DIM0] << ", " << outTensorDescs.at(1).shape.dims[DIM1]
                           << ", " << outTensorDescs.at(1).shape.dims[DIM2]);
        } else if (inTensorDescs.at(0).shape.dimNum == DIM2) {
            ATB_SPEED_LOG_DEBUG("[input0 dimNum = 2] CHECK W8A16_OP inputs shape: [input0]"
                           << inTensorDescs.at(0).shape.dims[DIM0] << ", "
                           << inTensorDescs.at(0).shape.dims[DIM1]);
            ATB_SPEED_LOG_DEBUG("[input0 dimNum = 2] CHECK W8A16_OP inputs shape: [input1]"
                           << inTensorDescs.at(1).shape.dims[DIM0] << ", "
                           << inTensorDescs.at(1).shape.dims[DIM1]);
            outTensorDescs.at(i).shape.dims[DIM0] = inTensorDescs.at(0).shape.dims[DIM0];
            outTensorDescs.at(i).shape.dims[DIM1] = inTensorDescs.at(0).shape.dims[DIM1];
            ATB_SPEED_LOG_DEBUG("[output0 dimNum = 2] CHECK W8A16_OP inputs shape: [output0]"
                           << outTensorDescs.at(0).shape.dims[DIM0] << ", "
                           << outTensorDescs.at(0).shape.dims[DIM1]);
            ATB_SPEED_LOG_DEBUG("[output1 dimNum = 2] CHECK W8A16_OP inputs shape: [output1]"
                           << outTensorDescs.at(1).shape.dims[DIM0] << ", "
                           << outTensorDescs.at(1).shape.dims[DIM1]);
        } else {
            ATB_SPEED_LOG_ERROR(opName_ << " invalid dim num:" << inTensorDescs.at(DIM0).shape.dimNum);
        }
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return 0;
}

uint32_t RmsNormOperation::GetInputNum() const { return NUM2; }

uint32_t RmsNormOperation::GetOutputNum() const { return NUM2; }

atb::Status RmsNormOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclTensor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(variantPack.inTensors.size());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        aclnnVariantPack.aclInTensors[i] = CreateTensor(variantPack.inTensors.at(i), i);
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " Create aclInTensor end");

    aclnnVariantPack.aclOutTensors.resize(variantPack.outTensors.size());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        aclnnVariantPack.aclOutTensors[i] = CreateTensor(variantPack.outTensors.at(i), i);
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " Create aclOutTensor end");
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclTensor end");
    return 0;
}

int RmsNormOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnRmsNormGetWorkspaceSize start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnRmsNormGetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor,
        aclnnVariantPack.aclInTensors.at(1)->tensor,
        this->epsilon,
        aclnnVariantPack.aclOutTensors.at(0)->tensor,
        aclnnVariantPack.aclOutTensors.at(1)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnRmsNormGetWorkspaceSize end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize << ", aclExecutor:"
                  << this->aclnnOpCache_->aclExecutor);

    return ret;
}

int RmsNormOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnRmsNorm start");
    int ret = aclnnRmsNorm(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnRmsNorm end, ret:" << ret);
    return ret;
}

std::shared_ptr<AclNNTensor> RmsNormOperation::CreateTensor(atb::Tensor atbTensor, int tensorIdx) const
{
    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = atbTensor;
    aclnnTensor->tensorIdx = tensorIdx;
    aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
    CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensor);
    return aclnnTensor;
}
} // namespace common
} // namespace atb_speed