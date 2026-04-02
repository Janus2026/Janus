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
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v2.h"
#include "atb/types.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "attn_operation.h"

namespace atb_speed {
namespace common {

AttnOperation::AttnOperation(const std::string &name, AclNNAttnParam param) : AclNNOperation(name), param_(param)
{
    tensorsOfValue[0] = nullptr;
    tensorsOfKey[0] = nullptr;
}

AttnOperation::~AttnOperation()
{
    tensorsOfKey[0] = nullptr;
    tensorsOfValue[0] = nullptr;
}

atb::Status AttnOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                      atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
    // FAtranslated [B,S,H], PA [B,S,N,D]
    outTensorDescs.at(0) = inTensorDescs.at(0);
    if (!param_.isFA) {
        outTensorDescs.at(0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0]; // B
        outTensorDescs.at(0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM2]; // N
        outTensorDescs.at(0).shape.dims[DIM2] = inTensorDescs.at(DIM0).shape.dims[DIM3]; // D
        outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum - 1;
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return 0;
}

uint32_t AttnOperation::GetInputNum() const
{
    uint32_t inputNum = 6;
    if (param_.hasKVQuant) {
        ++inputNum;
        if (param_.hasQuantOffset) {
            ++inputNum;
        }
    }
    return inputNum;
}

uint32_t AttnOperation::GetOutputNum() const { return NUM1; }

const int ACLNN_TENSOR_INDEX[8] = {0, 0, 0, 4, 6, 14, 12, 13};
const int ACLNN_TENSOR_LIST_INDEX[8] = {-1, 1, 2, -1, -1, -1, -1, -1};

atb::Status AttnOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        if (param_.isFA && i == 5) { // 5: idx of block tables in in tensor
            aclnnVariantPack.aclInTensors[i] = aclnnTensor;
            continue;
        }
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        aclnnTensor->tensorIdx = ACLNN_TENSOR_INDEX[i];
        aclnnTensor->tensorListidx = ACLNN_TENSOR_LIST_INDEX[i];
        if (i == NUM4) { // 4:actual seqLength index
            aclnnTensor->needUpdateTensorDataPtr = false;
            aclnnTensor->intArrayHostData.dataSize = aclnnTensor->atbTensor.dataSize / NUM4; // int32 has 4 bytes
            aclnnTensor->intArrayHostData.data.resize(aclnnTensor->intArrayHostData.dataSize);
            aclnnTensor->intArrayHostData.dataOri.resize(aclnnTensor->intArrayHostData.dataSize);
            std::transform(
                static_cast<int32_t *>(aclnnTensor->atbTensor.hostData),
                static_cast<int32_t *>(aclnnTensor->atbTensor.hostData) + aclnnTensor->atbTensor.dataSize / NUM4,
                aclnnTensor->intArrayHostData.data.data(), [](int32_t value) {
                    return static_cast<int64_t>(value);
            });
            std::copy(static_cast<int32_t *>(aclnnTensor->atbTensor.hostData),
                static_cast<int32_t *>(aclnnTensor->atbTensor.hostData) +
                    aclnnTensor->atbTensor.dataSize / sizeof(int32_t),
                aclnnTensor->intArrayHostData.dataOri.data());
            aclnnTensor->intArrayHostData.intArray = aclCreateIntArray(
                static_cast<int64_t *>(aclnnTensor->intArrayHostData.data.data()),
                aclnnTensor->intArrayHostData.dataSize);
        } else if (i == 3 && !param_.isFA) { // 3: idx of mask tensor
            aclnnTensor->needUpdateTensorDataPtr = false;
        } else {
            aclnnTensor->needUpdateTensorDataPtr = true;
            atb::Tensor atbTensor = variantPack.inTensors.at(i);
            aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
            CHECK_OPERATION_STATUS_RETURN(
                CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensor));
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    tensorsOfKey[0] = aclnnVariantPack.aclInTensors.at(1)->tensor;   // 1: key tensor index
    tensorsOfValue[0] = aclnnVariantPack.aclInTensors.at(2)->tensor; // 2: value tensor index
    auto tensorKeyList = aclCreateTensorList(tensorsOfKey, 1);
    auto tensorValueList = aclCreateTensorList(tensorsOfValue, 1);
    aclnnVariantPack.aclInTensorList.clear();
    aclnnVariantPack.aclInTensorList.push_back(nullptr);
    aclnnVariantPack.aclInTensorList.push_back(tensorKeyList);
    aclnnVariantPack.aclInTensorList.push_back(tensorValueList);
    return atb::NO_ERROR;
}

atb::Status AttnOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(NUM1);
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->tensorListidx = AclNNTensor::notInTensorList;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        atb::Tensor squeezedAtbTensor = variantPack.outTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(squeezedAtbTensor.desc.shape, squeezedAtbTensor.desc.shape,
                                                          squeezedAtbTensor, aclnnTensor));
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int AttnOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAttnGetWorkspaceSize start");
    char inputLayoutFA[5] = "BSH";
    char inputLayoutPA[5] = "BSND";
    double scaleValue = 1 / sqrt(param_.headDim);
    AclNNVariantPack &task = this->aclnnOpCache_->aclnnVariantPack;
    aclTensor *maskTensor = param_.isFA ? task.aclInTensors.at(3)->tensor : nullptr;  // 3: attention mask tensor index
    aclTensor *blockTensor = param_.isFA ? nullptr : task.aclInTensors.at(5)->tensor; // 5: blocktable tensor index
    aclTensor *antiquantScaleTensor =
        param_.hasKVQuant ? task.aclInTensors.at(6)->tensor : nullptr;                // 6: dequantOffset tensor index
    aclTensor *antiquantOffsetTensor = param_.hasKVQuant && param_.hasQuantOffset ? task.aclInTensors.at(7)->tensor
                                                                                  : nullptr; // 7: dequantOffset index
    // query - 0; key - 1; value - 2; pseShift - 3; attenMask - 4; actualSeqLengths - 5;
    // ++1 actualSeqLengthsKv - 6;
    // dequantScale1 - 6; quantScale1 - 7; dequantScale2 - 8; quantScale2 - 9;
    // quantScale2 - 10; antiquantScale - 11; antiquantOffset - 12; blocktable - 13;
    // numHeads - 14; scaleValue - 15; inputLayout - 16; numKeyValueHeads - 17;
    // blockSize - 18; innerPrecise - 19;
    // innerPrecise - 20; workspaceSize - 21; workspaceSize - 22
    int ret = aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
        task.aclInTensors.at(0)->tensor,        // 0: query index
        task.aclInTensorList.at(1),             // 1: key cache index
        task.aclInTensorList.at(2),             // 2: value cache index
        nullptr, maskTensor,                    // 4: attenMask
        nullptr, task.aclInTensors.at(4)->intArrayHostData.intArray, // 6: seq length index
        nullptr, nullptr, nullptr, nullptr, nullptr,
        antiquantScaleTensor,                   // 12: antiquantScale
        antiquantOffsetTensor,                  // 13: antiquantOffset
        blockTensor,                            // 14: blocktable
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        param_.headNum, scaleValue, 2147483647, 2147483647,
        param_.isFA ? inputLayoutFA : inputLayoutPA,
        param_.kvHeadNum, 0, param_.innerPrecise,
        param_.isFA ? 0 : param_.blockSize, 0, false, 0, 0,
        task.aclOutTensors.at(0)->tensor,      // 0: out tensor
        nullptr, &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAttnGetWorkspaceSize end, ret:" << ret <<
                        ", workspaceSize:" << this->aclnnOpCache_->workspaceSize <<
                        ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int AttnOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    return aclnnFusedInferAttentionScoreV2(workspace, this->aclnnOpCache_->workspaceSize,
                                           this->aclnnOpCache_->aclExecutor, stream);
}
} // namespace common
} // namespace atb_speed
