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
#include "aclnnop/aclnn_add_rms_norm_dynamic_quant.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "add_rms_norm_dynamic_quant_operation.h"

namespace atb_speed {
namespace common {

const double EPSILON_THRESHOLD = 1e-9; // translated

AddRmsNormDynamicQuantOperation::AddRmsNormDynamicQuantOperation(
    const std::string &name, double epsilon) : AclNNOperation(name)
{
    opName_ = name;
    if (std::abs(epsilon) > EPSILON_THRESHOLD) {
        epsilon_ = epsilon;
    }
}

AddRmsNormDynamicQuantOperation::~AddRmsNormDynamicQuantOperation()
{
    ATB_SPEED_LOG_DEBUG(opName_ << "AddRmsNormDynamicQuantOperation deconstructor");
    this->DestroyOperation();
}

atb::Status AddRmsNormDynamicQuantOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "AddRmsNormDynamicQuantOperation infer shape start");
    for (size_t i = 0; i < outTensorDescs.size(); i++) {
        outTensorDescs.at(i).format = inTensorDescs.at(0).format;
        if (i == 0 || i == NUM1) {  // y1Out,y2OuttranslateddtypetranslatedINT8
            outTensorDescs.at(i).dtype = aclDataType::ACL_INT8;
        } else if (i == NUM3 || i == NUM4) {  // scale1Out,scale2Out:FLOAT32
            outTensorDescs.at(i).dtype = aclDataType::ACL_FLOAT;
        } else {  // xOuttranslatedx1translateddtype
            outTensorDescs.at(i).dtype = inTensorDescs.at(0).dtype;
        }
        // translated smoothScaletranslated
        // y2Out,scale2Out translated1translated, translated, translated
        if (i == NUM1 || i == NUM4) {
            outTensorDescs.at(i).shape.dimNum = NUM1;
            outTensorDescs.at(i).shape.dims[0] = 1;
        } else if (i < NUM3) {
            // y1Out,xOuttranslated2-8translated, shape translatedx1, x2
            outTensorDescs.at(i).shape.dimNum = inTensorDescs.at(1).shape.dimNum;
            for (size_t j = 0; j < outTensorDescs.at(i).shape.dimNum; j++) {
                outTensorDescs.at(i).shape.dims[j] = inTensorDescs.at(1).shape.dims[j];
            }
        } else {
            // scale1Out:shapetranslatedxtranslatedshapetranslated
            outTensorDescs.at(i).shape.dimNum = inTensorDescs.at(1).shape.dimNum - 1;
            for (size_t j = 0; j < outTensorDescs.at(i).shape.dimNum; j++) {
                outTensorDescs.at(i).shape.dims[j] = inTensorDescs.at(1).shape.dims[j];
            }
        }
    }

    ATB_SPEED_LOG_DEBUG(opName_ << "AddRmsNormDynamicQuantOperation infer shape end");
    return 0;
}

uint32_t AddRmsNormDynamicQuantOperation::GetInputNum() const { return NUM3; }

uint32_t AddRmsNormDynamicQuantOperation::GetOutputNum() const { return NUM5; }

int AddRmsNormDynamicQuantOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclTensor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    // Create aclInTensor
    aclnnVariantPack.aclInTensors.resize(variantPack.inTensors.size());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        aclnnVariantPack.aclInTensors[i] = CreateTensor(variantPack.inTensors.at(i), i);
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " Create aclInTensor end");
    // Create aclOutTensor
    aclnnVariantPack.aclOutTensors.resize(variantPack.outTensors.size());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        aclnnVariantPack.aclOutTensors[i] = CreateTensor(variantPack.outTensors.at(i), i);
    }
    ATB_SPEED_LOG_DEBUG(opName_ << "Create aclOutTensor end; CreateAclTensor end");
    return 0;
}

int AddRmsNormDynamicQuantOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormQuantGetWorkspaceSize start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    // translated smoothScaletranslated
    int ret = aclnnAddRmsNormDynamicQuantGetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor,  // x1
        aclnnVariantPack.aclInTensors.at(1)->tensor,  // x2
        aclnnVariantPack.aclInTensors.at(2)->tensor,  // gamma(weight)
        nullptr,  // smoothScale1Optional
        nullptr,  // smoothScale2Optional
        epsilon_,  // epsilonOptional
        aclnnVariantPack.aclOutTensors.at(0)->tensor,  // y1Out
        aclnnVariantPack.aclOutTensors.at(1)->tensor,  // y2Out, shapetranslated1, translated, translated
        aclnnVariantPack.aclOutTensors.at(2)->tensor,  // xOut
        aclnnVariantPack.aclOutTensors.at(3)->tensor,  // scale1Out
        aclnnVariantPack.aclOutTensors.at(4)->tensor,  // scale2Out, shapetranslated1, translated, translated
        &this->aclnnOpCache_->workspaceSize,  // workspaceSize
        &this->aclnnOpCache_->aclExecutor);   // executor
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormQuantGetWorkspaceSize end, ret:" << ret
        << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize << ", aclExecutor:"
        << this->aclnnOpCache_->aclExecutor);

    return ret;
}

int AddRmsNormDynamicQuantOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormDynamicQuant start");
    int ret = aclnnAddRmsNormDynamicQuant(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormDynamicQuant end, ret:" << ret);
    return ret;
}

std::shared_ptr<AclNNTensor> AddRmsNormDynamicQuantOperation::CreateTensor(atb::Tensor atbTensor, int tensorIdx) const
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
