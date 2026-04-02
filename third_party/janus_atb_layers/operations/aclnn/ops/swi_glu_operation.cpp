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
#include "aclnnop/aclnn_swi_glu.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "swi_glu_operation.h"

namespace atb_speed {
namespace common {

SwiGluOperation::SwiGluOperation(const std::string &name, long dim) : AclNNOperation(name)
{
    this->opName_ = name;
    this->dim = dim;
}

atb::Status SwiGluOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                            atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
    ATB_SPEED_LOG_DEBUG("inTensorDescs.at(0).shape.dimNum" << inTensorDescs.at(0).shape.dimNum);
    
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    auto inTensorShape = inTensorDescs.at(0).shape;
    std::copy(inTensorShape.dims, inTensorShape.dims + inTensorShape.dimNum, outTensorDescs.at(0).shape.dims);
    outTensorDescs.at(0).shape.dimNum = inTensorShape.dimNum;

    auto dim = this->dim;
    // support reverse indexing
    if (dim < 0){
        dim += inTensorDescs.at(0).shape.dimNum;
    }

    if (dim < outTensorDescs.at(0).shape.dimNum) {
        outTensorDescs.at(0).shape.dims[dim] /= 2;
    } else {
        ATB_SPEED_LOG_ERROR("swiglu dim out of range, swiglu dim:  " << dim 
                << ",x dim = " << outTensorDescs.at(0).shape.dimNum);
    }
    ATB_SPEED_LOG_DEBUG("outTensorDescs.at(0).shape.dimNum" << outTensorDescs.at(0).shape.dimNum);
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return 0;
}

uint32_t SwiGluOperation::GetInputNum() const { return NUM1; }

uint32_t SwiGluOperation::GetOutputNum() const { return NUM1; }

atb::Status SwiGluOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
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

int SwiGluOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnSwiGluGetWorkspaceSize start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;

    auto dim = this->dim;
    // support reverse indexing
    if (dim < 0){
        dim += aclnnVariantPack.aclInTensors.at(0)->atbTensor.desc.shape.dimNum;
    };

    // get workspace
    int ret = aclnnSwiGluGetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor,
        this->dim,
        aclnnVariantPack.aclOutTensors.at(0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnSwiGluGetWorkspaceSize end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize << ", aclExecutor:"
                  << this->aclnnOpCache_->aclExecutor);

    return ret;
}

int SwiGluOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnSwiGlu start");
    int ret = aclnnSwiGlu(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnSwiGlu end, ret:" << ret);
    return ret;
}

std::shared_ptr<AclNNTensor> SwiGluOperation::CreateTensor(atb::Tensor atbTensor, int tensorIdx) const
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
