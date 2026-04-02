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
#include "moe_distribute_dispatch_operation.h"
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <vector>
#include <algorithm>
#include <atb/types.h>
#include <atb/comm.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_distribute_dispatch.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {

MoeDistributeDispatchOperation::MoeDistributeDispatchOperation(
    const std::string &name, MoeDistributeDispatchParam param) : AclNNOperation(name), param_(param) {}
MoeDistributeDispatchOperation::~MoeDistributeDispatchOperation() {}

atb::Status MoeDistributeDispatchOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "MoeDistributeDispatchOperation infer shape start");

    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = param_.isQuant ? aclDataType::ACL_INT8 : inTensorDescs.at(DIM0).dtype;
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;

    outTensorDescs.at(DIM1).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM1).dtype = aclDataType::ACL_FLOAT;
    outTensorDescs.at(DIM1).shape.dimNum = DIM1;

    outTensorDescs.at(DIM2).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM2).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(DIM2).shape.dimNum = DIM1;

    outTensorDescs.at(DIM3).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM3).dtype = aclDataType::ACL_INT64;
    outTensorDescs.at(DIM3).shape.dimNum = DIM1;

    outTensorDescs.at(NUM4).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(NUM4).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(NUM4).shape.dimNum = DIM1;

    outTensorDescs.at(NUM5).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(NUM5).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(NUM5).shape.dimNum = DIM1;

    outTensorDescs.at(NUM6).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(NUM6).dtype = aclDataType::ACL_FLOAT;
    outTensorDescs.at(NUM6).shape.dimNum = DIM1;

    ATB_SPEED_LOG_DEBUG(opName_
                  << "MoeDistributeDispatchOperation infer shape origin inTensorDescs.at(DIM0).shape.dims[DIM0]"
                  << inTensorDescs.at(DIM0).shape.dims[DIM0]);

    int32_t globalBS = GetGlobalBS(inTensorDescs.at(NUM3));
    int32_t globalTokenNum = globalBS * std::min(param_.localMoeExpertNum, param_.topk);

    outTensorDescs.at(DIM0).shape.dims[DIM0] = param_.epRankId < param_.sharedExpertRankNum ? \
        globalTokenNum / param_.sharedExpertRankNum : globalTokenNum; // translatedmmtranslated
    outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];

    outTensorDescs.at(DIM1).shape.dims[DIM0] =
        param_.epRankId < param_.sharedExpertRankNum ? globalTokenNum / param_.sharedExpertRankNum : globalTokenNum;

    outTensorDescs.at(DIM2).shape.dims[DIM0] = inTensorDescs.at(DIM1).shape.dims[DIM0] * \
        inTensorDescs.at(DIM1).shape.dims[DIM1];

    outTensorDescs.at(DIM3).shape.dims[DIM0] = param_.localMoeExpertNum;

    outTensorDescs.at(NUM4).shape.dims[DIM0] = param_.epRankSize * param_.localMoeExpertNum + \
        globalBS * param_.topk * (param_.epRankSize / NUM8) * NUM2;

    outTensorDescs.at(NUM5).shape.dims[DIM0] = 1;

    outTensorDescs.at(NUM6).shape.dims[DIM0] =
        param_.epRankId < param_.sharedExpertRankNum ? globalTokenNum / param_.sharedExpertRankNum : globalTokenNum;

    ATB_SPEED_LOG_DEBUG(opName_ << "MoeDistributeDispatchOperation infer shape end");
    return 0;
}

uint32_t MoeDistributeDispatchOperation::GetInputNum() const
{
    if (param_.quantSmooth) {
        return NUM5;
    } else {
        return NUM4;
    }
}

uint32_t MoeDistributeDispatchOperation::GetOutputNum() const
{
    return NUM7;
}

int32_t MoeDistributeDispatchOperation::GetGlobalBS(const atb::TensorDesc &inTensorDesc) const
{
    int32_t worldSize = param_.epRankSize * std::max(param_.tpRankSize, 1);
    if (param_.globalBS > 0) {
        return param_.globalBS;
    }
    int32_t maxDecodeDpTokenSize = param_.maxDecodeDpTokenSize;
    // if param_.maxDecodeDpTokenSize is not available,use in_padding_idx's DIM0
    if (maxDecodeDpTokenSize == 0) {
        maxDecodeDpTokenSize = inTensorDesc.shape.dims[DIM0];
    }
    return maxDecodeDpTokenSize * worldSize;
}

int MoeDistributeDispatchOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeDistributeDispatchOperation start");
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeDistributeDispatchOperation create hcclComm");
    ATB_SPEED_LOG_DEBUG("param_.epCommName " << param_.epCommName <<  "param_.tpCommName "  << param_.tpCommName
        << "param_.epRankSize " << param_.epRankSize
        << " param_.tpRankSize " << param_.tpRankSize
        << " param_.epRankId " << param_.epRankId
        << " param_.tpRankId " << param_.tpRankId
        << " param_.expertSharedType " << param_.expertSharedType
        << " param_.sharedExpertRankNum " << param_.sharedExpertRankNum << " param_.moeExpertNum "
        << param_.moeExpertNum << "param_.quantMode " << param_.quantMode
        << " param_.globalBS " << param_.globalBS);

    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;

    aclnnVariantPack.aclInTensors.at(NUM2)->tensorIdx = NUM4;
    aclnnVariantPack.aclInTensors.at(NUM3)->needUpdateTensorDataPtr = false;
    int32_t globalBS = GetGlobalBS(aclnnVariantPack.aclInTensors.at(NUM3)->atbTensor.desc);
    int ret = aclnnMoeDistributeDispatchGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclInTensors.at(DIM1)->tensor,
        param_.quantSmooth ? aclnnVariantPack.aclInTensors.at(DIM2)->tensor : nullptr,
        nullptr,
        aclnnVariantPack.aclInTensors.at(NUM2)->tensor,
        param_.epCommName.data(),
        param_.epRankSize,
        param_.epRankId,
        param_.moeExpertNum,
        param_.tpCommName.data(),
        param_.tpRankSize,
        param_.tpRankId,
        param_.expertSharedType,
        1,
        param_.sharedExpertRankNum,
        param_.quantMode,
        globalBS,
        param_.expertTokenNumsType,
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclOutTensors.at(DIM1)->tensor,
        aclnnVariantPack.aclOutTensors.at(DIM2)->tensor,
        aclnnVariantPack.aclOutTensors.at(NUM3)->tensor,
        aclnnVariantPack.aclOutTensors.at(NUM4)->tensor,
        aclnnVariantPack.aclOutTensors.at(NUM5)->tensor,
        aclnnVariantPack.aclOutTensors.at(NUM6)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int MoeDistributeDispatchOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeDistributeDispatchOperation start");

    int ret = aclnnMoeDistributeDispatch(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeDistributeDispatchOperation end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed