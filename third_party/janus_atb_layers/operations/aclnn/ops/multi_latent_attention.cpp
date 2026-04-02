#include "multi_latent_attention.h"
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "aclnn_multi_latent_attention.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {
MultiLatentAttentionOperation::MultiLatentAttentionOperation(
    const std::string &name,
    AclNNMultiLatentAttentionParam param
) : AclNNOperation(name), param_(param){}

MultiLatentAttentionOperation::~MultiLatentAttentionOperation() {
}

atb::Status MultiLatentAttentionOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "MultiLatentAttentionOperation infer shape start");
    outTensorDescs.at(DIM0) = inTensorDescs.at(DIM0);
    outTensorDescs.at(DIM0).dtype = inTensorDescs.at(DIM1).dtype;
    if (param_.calcType == AclNNMultiLatentAttentionParam::CalcType::CALC_TYPE_RING){
        outTensorDescs.at(DIM1) = outTensorDescs.at(DIM0);
        outTensorDescs.at(DIM1).shape.dims[DIM2] = 1;
    }
    ATB_SPEED_LOG_DEBUG(opName_ << "MultiLatentAttentionOperation infer shape end");
    return 0;
}

uint32_t MultiLatentAttentionOperation::GetInputNum() const 
{
    uint32_t intensorNumBase = IN_TENSOR_NUM;
    if (param_.maskType != AclNNMultiLatentAttentionParam::MaskType::UNDEFINED) {
        intensorNumBase++;
    }
    if (param_.calcType == AclNNMultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC) {
        intensorNumBase++;
    }
    if (param_.cacheMode == AclNNMultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        intensorNumBase += 2; // 2: qDescale kDescale
    }
    return intensorNumBase;
}

uint32_t MultiLatentAttentionOperation::GetOutputNum() const
{
    return param_.calcType != AclNNMultiLatentAttentionParam::CalcType::CALC_TYPE_RING ?
    OUT_TENSOR_NUM_1 : OUT_TENSOR_NUM_2;
}

atb::Status MultiLatentAttentionOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    uint32_t inTensorCount = aclnnVariantPack.aclInTensors.size();
    
    // Required Input Tensors
    for (size_t i = 0; i < this->IN_TENSOR_NUM; ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.inTensors.at(i), i);
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }

    uint32_t idx = this->IN_TENSOR_NUM;
    // Optional Input Tensors
    if (param_.maskType != AclNNMultiLatentAttentionParam::MaskType::UNDEFINED) {
        std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.inTensors.at(idx), 6);
        aclnnVariantPack.aclInTensors[idx++] = aclnnTensor;
    }

    if (param_.calcType == AclNNMultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC) {
        std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.inTensors.at(idx), 7);
        aclnnVariantPack.aclInTensors[idx++] = aclnnTensor;
    }

    if (param_.cacheMode == AclNNMultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        std::shared_ptr<AclNNTensor> aclnnTensorQDescale = CreateTensor(variantPack.inTensors.at(idx), 8);
        aclnnVariantPack.aclInTensors[idx++] = aclnnTensorQDescale;
        std::shared_ptr<AclNNTensor> aclnnTensorKDescale = CreateTensor(variantPack.inTensors.at(idx), 9);
        aclnnVariantPack.aclInTensors[idx++] = aclnnTensorKDescale;
    }
    
    if (idx != inTensorCount) {
        ATB_SPEED_LOG_ERROR(opName_ << " CreateAclNNInTensorVariantPack failed, tensor count not match, expect:"
                    << inTensorCount << ", actual:" << idx);
        return atb::ERROR_INTERNAL_ERROR;
    }

    int qSeqIdx = this->QSEQLEN_INDEX;
    if (param_.maskType != AclNNMultiLatentAttentionParam::MaskType::UNDEFINED) {
        qSeqIdx++;
    }
    int ret = this->BuildFromTensor(variantPack, this->CONTEXTLENS_INDEX, qSeqIdx, param_.calcType == AclNNMultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(opName_ << " BuildFromTensor failed");
        return atb::ERROR_INTERNAL_ERROR;
    }
    return atb::NO_ERROR;
}

atb::Status MultiLatentAttentionOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.outTensors.at(i), i);
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int MultiLatentAttentionOperation::SetAclNNWorkspaceExecutor()
{
    uint32_t inputNum = this->GetInputNum();
    uint32_t outputNum = this->GetOutputNum();
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;

    // Get all input and output tensors here to put them into correct order
    // which may be different due to optional tensors

    // The first 6 tensors are required
    aclTensor *query = aclnnVariantPack.aclInTensors.at(DIM0)->tensor;
    aclTensor *queryRope = aclnnVariantPack.aclInTensors.at(DIM1)->tensor;
    aclTensor *kvCache = aclnnVariantPack.aclInTensors.at(DIM2)->tensor;
    aclTensor *kvCacheRope = aclnnVariantPack.aclInTensors.at(DIM3)->tensor;
    aclTensor *blockTables = aclnnVariantPack.aclInTensors.at(4)->tensor;
    aclTensor *contextLens = aclnnVariantPack.aclInTensors.at(5)->tensor;
    // Optional tensors
    uint32_t idx = this->IN_TENSOR_NUM;
    aclTensor *mask = param_.maskType != AclNNMultiLatentAttentionParam::MaskType::UNDEFINED ?
        aclnnVariantPack.aclInTensors.at(idx++)->tensor : nullptr;
    aclTensor *qSeqlen = param_.calcType == AclNNMultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC ?
        aclnnVariantPack.aclInTensors.at(idx++)->tensor : nullptr;
    aclTensor *qkDescale = param_.cacheMode == AclNNMultiLatentAttentionParam::CacheMode::INT8_NZCACHE ?
        aclnnVariantPack.aclInTensors.at(idx++)->tensor : nullptr;
    aclTensor *pvDescale = param_.cacheMode == AclNNMultiLatentAttentionParam::CacheMode::INT8_NZCACHE ?
        aclnnVariantPack.aclInTensors.at(idx++)->tensor : nullptr;
    
    if (idx != inputNum) {
        ATB_SPEED_LOG_ERROR(opName_ << " SetAclNNWorkspaceExecutor failed, tensor count not match, expect:" 
                    << inputNum << ", actual:" << idx);
        return 1;
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor inputNum:" << inputNum
                  << ", outputNum:" << outputNum
                  << ", mask:" << (mask != nullptr)
                  << ", qSeqlen:" << (qSeqlen != nullptr)
                  << ", qDescale:" << (qkDescale != nullptr)
                  << ", kDescale:" << (pvDescale != nullptr));


    int ret = aclnnMultiLatentAttentionGetWorkspaceSize(
        query,
        queryRope,
        kvCache,
        kvCacheRope,
        blockTables,
        contextLens,
        mask,
        qSeqlen,
        qkDescale,
        pvDescale,
        0, // only support type = 0, which means split cache
        param_.headNum,
        param_.qkScale,
        param_.kvHeadNum,
        param_.maskType,
        param_.calcType == AclNNMultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC ? 
        aclCreateIntArray(reinterpret_cast<int64_t *>(this->qSeqLen.data()), this->qSeqLen.size()) : nullptr,
        // kv seqlen TODO
        aclCreateIntArray(reinterpret_cast<int64_t *>(this->kvSeqLen.data()), this->kvSeqLen.size()),
        param_.calcType == AclNNMultiLatentAttentionParam::CalcType::CALC_TYPE_RING,
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        outputNum > 1 ? aclnnVariantPack.aclOutTensors.at(DIM1)->tensor : nullptr,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);

    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int MultiLatentAttentionOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnMultiLatentAttention start");
    int ret = aclnnMultiLatentAttention(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnMultiLatentAttention end, ret:" << ret);
    return ret;
}

int MultiLatentAttentionOperation::BuildFromTensor(const atb::VariantPack &variantPack, size_t contextLensTensorId, size_t qSeqlenTensorId, bool needQLens)
{
    this->qSeqLen.clear();
    this->kvSeqLen.clear();
    atb::Tensor contextLensTensor = variantPack.inTensors.at(contextLensTensorId);
    if (contextLensTensor.desc.dtype != aclDataType::ACL_INT32) {
        ATB_SPEED_LOG_ERROR(opName_ << "contextLensTensor dtype is not int32");
        return 1;
    }
    if (!contextLensTensor.hostData) {
        ATB_SPEED_LOG_ERROR(opName_ << "contextLensTensor.hostData is null");
        return 1;
    }
    this->kvSeqLen.resize(contextLensTensor.dataSize / sizeof(int32_t));
    int32_t *contextLensTensorHostData = (int32_t *)contextLensTensor.hostData;
    for (size_t i = 0; i < this->kvSeqLen.size(); ++i) {
        this->kvSeqLen[i] = contextLensTensorHostData[i];
    }
    if (needQLens) {
        atb::Tensor qSeqlenTensor = variantPack.inTensors.at(qSeqlenTensorId);
         if (!qSeqlenTensor.hostData) {
            ATB_SPEED_LOG_ERROR(opName_ << "qSeqlenTensor.hostData is null");
            return false;
        }
        this->qSeqLen.resize(qSeqlenTensor.dataSize / sizeof(int32_t));
        int32_t *qSeqlenTensorHostData = (int32_t *)qSeqlenTensor.hostData;
        for (size_t i = 0; i < this->qSeqLen.size(); ++i) {
            this->qSeqLen[i] = qSeqlenTensorHostData[i];
        }
    }
    return 0;
}

}  // namespace common
}  // namespace atb_speed
