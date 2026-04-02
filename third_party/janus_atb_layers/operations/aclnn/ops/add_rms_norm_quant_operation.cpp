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
#include "aclnnop/aclnn_add_rms_norm_quant.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "add_rms_norm_quant_operation.h"

namespace atb_speed {
namespace common {

const double EPSILON_THRESHOLD = 1e-9; // translated

AddRmsNormQuantOperation::AddRmsNormQuantOperation(const std::string &name, double epsilon) : AclNNOperation(name)
{
    opName_ = name;
    if (std::abs(epsilon) > EPSILON_THRESHOLD) {
        epsilon_ = epsilon;
    }
}

AddRmsNormQuantOperation::~AddRmsNormQuantOperation()
{
    ATB_SPEED_LOG_DEBUG(opName_ << "AddRmsNormQuantOperation deconstructor");
    this->DestroyOperation();
}

atb::Status AddRmsNormQuantOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "AddRmsNormQuantOperation infer shape start");
    for (size_t i = 0; i < outTensorDescs.size(); i++) {
        outTensorDescs.at(i).format = inTensorDescs.at(0).format;
        if (i == 0 || i == NUM1) {  // y1Out,y2OuttranslateddtypetranslatedINT8
            outTensorDescs.at(i).dtype = aclDataType::ACL_INT8;
        } else {  // xOuttranslatedx1translateddtype
            outTensorDescs.at(i).dtype = inTensorDescs.at(0).dtype;
        }

        // translated1-8translated, shape translatedx1, x2
        outTensorDescs.at(i).shape.dimNum = inTensorDescs.at(1).shape.dimNum;
        for (size_t j = 0; j < outTensorDescs.at(i).shape.dimNum; j++) {
            outTensorDescs.at(i).shape.dims[j] = inTensorDescs.at(1).shape.dims[j];
        }
    }

    ATB_SPEED_LOG_DEBUG(opName_ << "AddRmsNormQuantOperation infer shape end");
    return 0;
}

uint32_t AddRmsNormQuantOperation::GetInputNum() const { return NUM5; }

uint32_t AddRmsNormQuantOperation::GetOutputNum() const { return NUM3; }

int AddRmsNormQuantOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclTensor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    // Create aclInTensor
    aclnnVariantPack.aclInTensors.resize(variantPack.inTensors.size());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor atbTensor = variantPack.inTensors.at(i);

        aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);

        atb::Dims viewDims = atbTensor.desc.shape;
        if (i == NUM4) {  // zeroPoints1Optional fp16translated:DT_INT32, bf16translated:DT_BFLOAT16
            // tensorIdxtranslated14translatedidxtranslated, itranslatedinTensors(5translated)translated;
            // translatedinTensorstranslatednullptrtranslated, translatedidxtranslateditranslated(translatedtensorIdxtranslated,translatednullptr)
            aclnnTensor->tensorIdx = NUM5;
        }
        aclnnTensor->tensor = aclCreateTensor(
            viewDims.dims, atbTensor.desc.shape.dimNum, atbTensor.desc.dtype,
            aclnnTensor->strides.data(), 0, atbTensor.desc.format, viewDims.dims,
            atbTensor.desc.shape.dimNum, atbTensor.deviceData);
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " Create aclInTensor end");
    // Create aclOutTensor
    aclnnVariantPack.aclOutTensors.resize(variantPack.outTensors.size());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        aclnnVariantPack.aclOutTensors[i] = CreateTensor(variantPack.outTensors.at(i), i);
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " Create aclOutTensor end; CreateAclTensor end");
    return 0;
}

int AddRmsNormQuantOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormQuantGetWorkspaceSize start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnAddRmsNormQuantGetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor,  // x1
        aclnnVariantPack.aclInTensors.at(1)->tensor,  // x2
        aclnnVariantPack.aclInTensors.at(2)->tensor,  // gamma(weight)
        aclnnVariantPack.aclInTensors.at(3)->tensor,  // scales1
        nullptr,  // scales2Optional  -> translated
        aclnnVariantPack.aclInTensors.at(4)->tensor,  // zeroPoints1Optional
        nullptr,  // zeroPoints2Optional  -> translated
        -1,
        epsilon_,  // epsilonOptional
        true,  // divMode
        aclnnVariantPack.aclOutTensors.at(0)->tensor,  // y1Out
        aclnnVariantPack.aclOutTensors.at(1)->tensor,  // y2Out, shapetranslated1, translated
        aclnnVariantPack.aclOutTensors.at(2)->tensor,  // xOut
        &this->aclnnOpCache_->workspaceSize,  // workspaceSize
        &this->aclnnOpCache_->aclExecutor);   // executor
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormQuantGetWorkspaceSize end, ret:" << ret
        << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize << ", aclExecutor:"
        << this->aclnnOpCache_->aclExecutor);

    return ret;
}

int AddRmsNormQuantOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormQuant start");
    int ret = aclnnAddRmsNormQuant(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormQuant end, ret:" << ret);
    return ret;
}

std::shared_ptr<AclNNTensor> AddRmsNormQuantOperation::CreateTensor(atb::Tensor atbTensor, int tensorIdx) const
{
    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = atbTensor;
    aclnnTensor->tensorIdx = tensorIdx;
    aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
    aclnnTensor->tensor = aclCreateTensor(
        atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum,
        atbTensor.desc.dtype, aclnnTensor->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims,
        atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    return aclnnTensor;
}
} // namespace common
} // namespace atb_speed