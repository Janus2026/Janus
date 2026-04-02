/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "lightning_indexer_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_lightning_indexer.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {

LightningIndexerOperation::LightningIndexerOperation(const std::string &name, LightningIndexerParam param)
    : AclNNOperation(name), param_(param)
{
}

LightningIndexerOperation::~LightningIndexerOperation()
{
    ATB_SPEED_LOG_DEBUG("LightningIndexerOperation deconstruct");
    this->DestroyOperation();
}

uint32_t LightningIndexerOperation::GetInputNum() const
{
    return 6; // query, key, weight, seq_len_query, seq_len_key, block_table
}

uint32_t LightningIndexerOperation::GetOutputNum() const { return 1; }

atb::Status LightningIndexerOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                                                  atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " LightningIndexerOperation infer shape start");

    outTensorDesc.at(0).format = inTensorDesc.at(0).format;
    outTensorDesc.at(0).dtype = ACL_INT32;
    if (param_.queryLayout == "BSND") {
        outTensorDesc.at(0).shape.dimNum = 4;
        outTensorDesc.at(0).shape.dims[0] = inTensorDesc.at(0).shape.dims[0];
        outTensorDesc.at(0).shape.dims[1] = inTensorDesc.at(0).shape.dims[1];
        outTensorDesc.at(0).shape.dims[2] = inTensorDesc.at(1).shape.dims[2];
        outTensorDesc.at(0).shape.dims[3] = param_.selectedCount;
    } else {
        outTensorDesc.at(0).shape.dimNum = 3;
        outTensorDesc.at(0).shape.dims[0] = inTensorDesc.at(0).shape.dims[0];
        outTensorDesc.at(0).shape.dims[1] = inTensorDesc.at(1).shape.dims[2];
        outTensorDesc.at(0).shape.dims[2] = param_.selectedCount;
    }
    ATB_SPEED_LOG_DEBUG(opName_ << "LightningIndexerOperation InferShape end");

    return atb::NO_ERROR;
}

atb::Status LightningIndexerOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.inTensors.at(i), i);
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

atb::Status LightningIndexerOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.outTensors.at(i), i);
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int LightningIndexerOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;

    atb::Tensor atbTensor = aclnnVariantPack.aclOutTensors.at(0)->atbTensor;
    aclTensor *tensor = aclCreateTensor(
        atbTensor.desc.shape.dims,
        atbTensor.desc.shape.dimNum,
        ACL_BF16,
        aclnnVariantPack.aclOutTensors.at(0)->strides.data(),
        0,
        atbTensor.desc.format,
        atbTensor.desc.shape.dims,
        atbTensor.desc.shape.dimNum,
        atbTensor.deviceData);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start_exec");
   
    int ret =
        aclnnLightningIndexerGetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor,    // query
                                              aclnnVariantPack.aclInTensors.at(1)->tensor,    // key
                                              aclnnVariantPack.aclInTensors.at(2)->tensor,    // weights
                                              aclnnVariantPack.aclInTensors.at(3)->tensor,    // query_seq_lengths
                                              aclnnVariantPack.aclInTensors.at(4)->tensor,    // key_seq_lengths
                                              aclnnVariantPack.aclInTensors.at(5)->tensor,    // block_table
                                              const_cast<char *>(param_.queryLayout.c_str()), // query_layout
                                              const_cast<char *>(param_.keyLayout.c_str()),   // key_layout
                                              param_.selectedCount, param_.sparseMode,
                                              param_.preTokens, param_.nextTokens, false,
                                              aclnnVariantPack.aclOutTensors.at(0)->tensor, // out
                                              tensor,                                      // sparseValuesOut
                                              &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end"
                                << ", ret: " << ret << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                                << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int LightningIndexerOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret =
        aclnnLightningIndexer(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end, ret:" << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed
