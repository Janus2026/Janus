/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "moe_distribute_combine_v2_operation.h"
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <atb/types.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_distribute_combine_v2.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {

MoeDistributeCombineV2Operation::MoeDistributeCombineV2Operation(
    const std::string &name, MoeDistributeCombineV2Param param) : AclNNOperation(name), param_(param) {}
MoeDistributeCombineV2Operation::~MoeDistributeCombineV2Operation() {}

atb::Status MoeDistributeCombineV2Operation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "MoeDistributeCombineV2Operation infer shape start");

    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = inTensorDescs.at(DIM0).dtype;
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;

    ATB_SPEED_LOG_DEBUG(opName_
                  << "MoeDistributeCombineV2Operation infer shape origin inTensorDescs.at(DIM1).shape.dims[DIM0]"
                  << inTensorDescs.at(DIM1).shape.dims[DIM0]);
    ATB_SPEED_LOG_DEBUG(opName_
                  << "MoeDistributeCombineV2Operation infer shape origin inTensorDescs.at(DIM0).shape.dims[DIM1]"
                  << inTensorDescs.at(DIM0).shape.dims[DIM1]);
    outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM1).shape.dims[DIM0];
    outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];
    ATB_SPEED_LOG_DEBUG(opName_ << "MoeDistributeCombineV2Operation infer shape end");
    return 0;
}
uint32_t MoeDistributeCombineV2Operation::GetInputNum() const
{
    return NUM7; // 7translatedintensor, translatedcombine_v1translated
}

uint32_t MoeDistributeCombineV2Operation::GetOutputNum() const
{
    return DIM1;
}

int32_t MoeDistributeCombineV2Operation::GetGlobalBS(const atb::TensorDesc &inTensorDesc) const
{
    int32_t worldSize = param_.epRankSize * std::max(param_.tpRankSize, 1);
    if (param_.globalBS > 0) {
        return param_.globalBS;
    }
    int32_t maxDecodeDpTokenSize = param_.maxDecodeDpTokenSize;
    // if param_.maxDecodeDpTokenSize is not available,use in_padding_idx's DIM0
    if (maxDecodeDpTokenSize == 0) {
        maxDecodeDpTokenSize = inTensorDesc.shape.dims[DIM0] / \
            std::min(param_.localMoeExpertNum, param_.topk) / worldSize;
    }
    return maxDecodeDpTokenSize * worldSize;
}

int MoeDistributeCombineV2Operation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeDistributeCombineV2Operation start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    
    aclnnVariantPack.aclInTensors.at(NUM6)->tensorIdx = NUM10;
    int64_t globalBS = GetGlobalBS(aclnnVariantPack.aclInTensors.at(DIM0)->atbTensor.desc);
    int ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclInTensors.at(DIM1)->tensor,
        aclnnVariantPack.aclInTensors.at(DIM2)->tensor,
        aclnnVariantPack.aclInTensors.at(NUM3)->tensor,
        aclnnVariantPack.aclInTensors.at(NUM4)->tensor,
        aclnnVariantPack.aclInTensors.at(NUM5)->tensor,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        aclnnVariantPack.aclInTensors.at(NUM6)->tensor,
        nullptr,
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
        globalBS,
        0,
        param_.commQuantMode,
        0,
        param_.commAlg.data(),
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int MoeDistributeCombineV2Operation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeDistributeCombineV2Operation start");

    int ret = aclnnMoeDistributeCombineV2(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeDistributeCombineV2Operation end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed