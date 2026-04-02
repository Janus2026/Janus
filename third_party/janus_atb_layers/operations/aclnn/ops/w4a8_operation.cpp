/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "operations/aclnn/core/acl_nn_operation.h"
#include "aclnnop/aclnn_quant_matmul_v5.h"
#include "w4a8_operation.h"

namespace atb_speed {
namespace common {

W4A8Operation::W4A8Operation(
    const std::string &name,
    AclNNW4A8Param param) : AclNNOperation(name), param_(param) {}

W4A8Operation::~W4A8Operation()
{
    ATB_SPEED_LOG_DEBUG("W4A8Operation deconstructor");
    this->DestroyOperation();
}

atb::Status W4A8Operation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                      atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
    outTensorDescs.at(0).format = inTensorDescs.at(0).format; // NDtranslatedNZ
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum; // tensortranslated
    outTensorDescs.at(0).dtype = param_.outDataType;
    if (inTensorDescs.at(0).shape.dimNum == DIM2) {
        ATB_SPEED_LOG_DEBUG("[input0 dimNum = 2] CHECK " << opName_ << " inputs shape: [input0]"
                       << inTensorDescs.at(DIM0).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM0).shape.dims[DIM1]);
        ATB_SPEED_LOG_DEBUG("[input0 dimNum = 2] CHECK " << opName_ << " inputs shape: [input1]"
                       << inTensorDescs.at(DIM1).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM1).shape.dims[DIM1]);
        outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        // 8: int4 packed int32
        outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM1).shape.dims[DIM1] * 8;
    } else {
        ATB_SPEED_LOG_ERROR(opName_ << " invalid dim num:" << inTensorDescs.at(DIM0).shape.dimNum);
        return atb::ERROR_INVALID_TENSOR_DIM_NUM;
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return atb::NO_ERROR;
}

uint32_t W4A8Operation::GetInputNum() const { return 5; }               // 5: x, weight, x_scale, weight_scale, y_offset

uint32_t W4A8Operation::GetOutputNum() const { return 1; }              // 1: y

int W4A8Operation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        atb::Tensor atbTensor = variantPack.inTensors.at(i);
        if (i == 2) {                                                   // 2: x_scale
            atbTensor.desc.shape.dimNum = 2;                            // 2: dimnum = 2, dims = [m, 1]
            atbTensor.desc.shape.dims[1] = 1;                           // 1: dims[1] = 1
        }
        int tensorIdx = i == 4 ? 7 : i;
        std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(atbTensor, tensorIdx);
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int W4A8Operation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnQuantMatmulV5GetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor,  // 0: input
        aclnnVariantPack.aclInTensors.at(1)->tensor,  // 1: weight
        aclnnVariantPack.aclInTensors.at(2)->tensor,  // 2: x1scale
        aclnnVariantPack.aclInTensors.at(3)->tensor,  // 3: x2scale
        nullptr,                // yscale
        nullptr,                // x1offset
        nullptr,                // x2offset
        aclnnVariantPack.aclInTensors.at(4)->tensor, // yoffset
        nullptr, // bias
        false, // transposeX1
        false, // transposeX2
        param_.groupSize, // groupSize
        aclnnVariantPack.aclOutTensors.at(0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " end, ret:"
                  << ret << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int W4A8Operation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    int ret = aclnnQuantMatmulV5(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR("ExecuteAclNNOp failed, ret: " << ret);
    }
    return ret;
}

} // namespace common
} // namespace atb_speed

