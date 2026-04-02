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
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "aclrt_cmo_async.h"

namespace atb_speed {
namespace common {

AclrtCmoAsyncOperation::AclrtCmoAsyncOperation(const std::string &opName, 
    double coefficient) : opName_(opName), coefficient_(coefficient) {}

AclrtCmoAsyncOperation::~AclrtCmoAsyncOperation()
{
    ATB_SPEED_LOG_DEBUG("AclrtCmoAsyncOperation deconstructor");
}

std::string AclrtCmoAsyncOperation::GetName() const
{
    return this->opName_;
}


uint32_t AclrtCmoAsyncOperation::GetInputNum() const
{
    return NUM1;
}

uint32_t AclrtCmoAsyncOperation::GetOutputNum() const
{
    return 0;
}

atb::Status AclrtCmoAsyncOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                                               atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
    ATB_SPEED_LOG_DEBUG("inTensorDesc size: " << inTensorDesc.size() << ", outTensorDesc size: "
                        << outTensorDesc.size());
    return atb::NO_ERROR;
}

atb::Status AclrtCmoAsyncOperation::Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize,
                                          atb::Context *context)
{
    ATB_SPEED_LOG_DEBUG("variantPack outTensors size: "
                        << variantPack.outTensors.size()
                        << ", workspaceSize: "
                        << workspaceSize);
    ATB_SPEED_LOG_DEBUG(this->opName_ << " setup start");

    if (context == nullptr) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " setup context is null");
        return atb::ERROR_INVALID_PARAM;
    }

    workspaceSize = 0;

    ATB_SPEED_LOG_DEBUG("setup end");
    return atb::NO_ERROR;
}

atb::Status AclrtCmoAsyncOperation::Execute(const atb::VariantPack &variantPack, uint8_t *workspace,
                                            uint64_t workspaceSize, atb::Context *context)
{
    ATB_SPEED_LOG_DEBUG(this->opName_ << " execute start: ");

    if (!context) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " execute fail, context param is null. Enable log: "
            << "export ASDOPS_LOG_LEVEL=ERROR, export ASDOPS_LOG_TO_STDOUT=1 to find the first error. "
            << "For more details, see the MindIE official document." << std::endl, ATB_MODELS_EXECUTION_FAILURE);
        return atb::ERROR_INVALID_PARAM;
    }

    std::vector<aclrtStream> streams = context->GetExecuteStreams();

    if (!streams[1]) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " execute fail, execute stream in context is null. "
            << "Enable log: export ASDOPS_LOG_LEVEL=ERROR, export ASDOPS_LOG_TO_STDOUT=1 to find the first error. "
            << "For more details, see the MindIE official document." << std::endl, ATB_MODELS_EXECUTION_FAILURE);
        return atb::ERROR_INVALID_PARAM;
    }

    aclrtCmoType cmoType = ACL_RT_CMO_TYPE_PREFETCH;

    ATB_SPEED_LOG_DEBUG("variantPack deviceData: " << variantPack.inTensors.at(0).deviceData
                        << " ,variantPack dataSize: " << variantPack.inTensors.at(0).dataSize
                        << " ,stream: " << streams[1]);

    CheckAcl(aclrtCmoAsync(variantPack.inTensors.at(0).deviceData,
                           variantPack.inTensors.at(0).dataSize * this->coefficient_,
                           cmoType,
                           streams[1]));

    ATB_SPEED_LOG_DEBUG("aclrtCmoAsync create success.");

    if (workspaceSize != 0 || workspace != nullptr) {
        ATB_SPEED_LOG_DEBUG("execute workspace: " << workspaceSize);
    }

    return atb::NO_ERROR;
}

aclError AclrtCmoAsyncOperation::CheckAcl(aclError ret) const
{
    if (ret != ACL_ERROR_NONE) {
        ATB_SPEED_LOG_ERROR(__FILE__ << ":" << __LINE__ << " aclError:" << ret);
    }
    return ret;
}

} // namespace common
} // namespace atb_speed
