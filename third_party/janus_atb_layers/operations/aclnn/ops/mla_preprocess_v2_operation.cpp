/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "mla_preprocess_v2_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "aclnn_mla_preprocess_v2.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {

Mlapreprocessv2Operation::Mlapreprocessv2Operation(const std::string &name,
                                                                 Mlapreprocessv2OperationParam param)
    : AclNNOperation(name), param_(param)
{
}

Mlapreprocessv2Operation::~Mlapreprocessv2Operation()
{
    ATB_SPEED_LOG_DEBUG("Mlapreprocessv2Operation deconstruct");
    this->DestroyOperation();
}

uint32_t Mlapreprocessv2Operation::GetInputNum() const
{
    return 24;
}

uint32_t Mlapreprocessv2Operation::GetOutputNum() const { return 5; }

atb::Status Mlapreprocessv2Operation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                                                        atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " Mlapreprocessv2Operation infer shape start");
    outTensorDesc.at(0).format = inTensorDesc.at(0).format;
    outTensorDesc.at(0).dtype = inTensorDesc.at(0).dtype;
    outTensorDesc.at(0).shape.dimNum = 3;
    outTensorDesc.at(0).shape.dims[0] = inTensorDesc.at(0).shape.dims[0];
    outTensorDesc.at(0).shape.dims[1] = inTensorDesc.at(18).shape.dims[0];
    outTensorDesc.at(0).shape.dims[2] = 512;

    outTensorDesc.at(1).format = inTensorDesc.at(0).format;
    outTensorDesc.at(1).dtype = inTensorDesc.at(0).dtype;
    outTensorDesc.at(1).shape.dimNum = 4;
    outTensorDesc.at(1).shape.dims[0] = inTensorDesc.at(19).shape.dims[0];
    outTensorDesc.at(1).shape.dims[1] = inTensorDesc.at(19).shape.dims[1];
    outTensorDesc.at(1).shape.dims[2] = inTensorDesc.at(19).shape.dims[2];
    outTensorDesc.at(1).shape.dims[3] = inTensorDesc.at(19).shape.dims[3];

    outTensorDesc.at(2).format = inTensorDesc.at(0).format;
    outTensorDesc.at(2).dtype = inTensorDesc.at(0).dtype;
    outTensorDesc.at(2).shape.dimNum = 3;
    outTensorDesc.at(2).shape.dims[0] = inTensorDesc.at(0).shape.dims[0];
    outTensorDesc.at(2).shape.dims[1] = inTensorDesc.at(18).shape.dims[0];
    outTensorDesc.at(2).shape.dims[2] = 64;

    outTensorDesc.at(3).format = inTensorDesc.at(0).format;
    outTensorDesc.at(3).dtype = inTensorDesc.at(0).dtype;
    outTensorDesc.at(3).shape.dimNum = 4;
    outTensorDesc.at(3).shape.dims[0] = inTensorDesc.at(20).shape.dims[0];
    outTensorDesc.at(3).shape.dims[1] = inTensorDesc.at(20).shape.dims[1];
    outTensorDesc.at(3).shape.dims[2] = inTensorDesc.at(20).shape.dims[2];
    outTensorDesc.at(3).shape.dims[3] = inTensorDesc.at(20).shape.dims[3];

    outTensorDesc.at(4).format = inTensorDesc.at(0).format;
    outTensorDesc.at(4).dtype = inTensorDesc.at(0).dtype;
    outTensorDesc.at(4).shape.dimNum = 2;
    outTensorDesc.at(4).shape.dims[0] = inTensorDesc.at(0).shape.dims[0];
    outTensorDesc.at(4).shape.dims[1] = inTensorDesc.at(8).shape.dims[0];
    ATB_SPEED_LOG_DEBUG(opName_ << "Mlapreprocessv2Operation InferShape end");

    return atb::NO_ERROR;
}

atb::Status Mlapreprocessv2Operation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.inTensors.at(i), i);
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

atb::Status Mlapreprocessv2Operation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.outTensors.at(i), i);
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int Mlapreprocessv2Operation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = 
        aclnnMlaPreprocessV2GetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor,
                                            aclnnVariantPack.aclInTensors.at(1)->tensor,
                                            aclnnVariantPack.aclInTensors.at(2)->tensor,
                                            aclnnVariantPack.aclInTensors.at(3)->tensor,
                                            aclnnVariantPack.aclInTensors.at(4)->tensor,
                                            aclnnVariantPack.aclInTensors.at(5)->tensor,
                                            aclnnVariantPack.aclInTensors.at(6)->tensor,
                                            aclnnVariantPack.aclInTensors.at(7)->tensor,
                                            aclnnVariantPack.aclInTensors.at(8)->tensor,
                                            aclnnVariantPack.aclInTensors.at(9)->tensor,
                                            aclnnVariantPack.aclInTensors.at(10)->tensor,
                                            aclnnVariantPack.aclInTensors.at(11)->tensor,
                                            aclnnVariantPack.aclInTensors.at(12)->tensor,
                                            aclnnVariantPack.aclInTensors.at(13)->tensor,
                                            aclnnVariantPack.aclInTensors.at(14)->tensor,
                                            aclnnVariantPack.aclInTensors.at(15)->tensor,
                                            aclnnVariantPack.aclInTensors.at(16)->tensor,
                                            aclnnVariantPack.aclInTensors.at(17)->tensor,
                                            aclnnVariantPack.aclInTensors.at(18)->tensor,
                                            aclnnVariantPack.aclInTensors.at(19)->tensor,
                                            aclnnVariantPack.aclInTensors.at(20)->tensor,
                                            aclnnVariantPack.aclInTensors.at(21)->tensor,
                                            aclnnVariantPack.aclInTensors.at(22)->tensor,
                                            aclnnVariantPack.aclInTensors.at(23)->tensor,
                                            param_.wdqDim,
                                            param_.qRopeDim,
                                            param_.kRopeDim,
                                            param_.epsilon,
                                            param_.qRotaryCoeff,
                                            param_.kRotaryCoeff,
                                            param_.transeposeWdq,
                                            param_.transeposeWuq,
                                            param_.transeposeWuk,
                                            param_.cacheMode,
                                            param_.quantMode,
                                            param_.doRmsNorm,
                                            param_.wdkvSplitCount, 
                                            param_.qDownOutFlag,
                                            aclnnVariantPack.aclOutTensors.at(0)->tensor,
                                            aclnnVariantPack.aclOutTensors.at(1)->tensor,
                                            aclnnVariantPack.aclOutTensors.at(2)->tensor,
                                            aclnnVariantPack.aclOutTensors.at(3)->tensor,
                                            aclnnVariantPack.aclOutTensors.at(4)->tensor, 
                                            &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);

    
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end"
                                << ", ret: " << ret << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                                << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int Mlapreprocessv2Operation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnMlaPreprocessV2(workspace, this->aclnnOpCache_->workspaceSize,
                                          this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end, ret:" << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed

