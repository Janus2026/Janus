/* Copyright 2025 The Janus Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at


Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <numeric>
#include <algorithm>
#include "multi_latent_attention_tiling_impl.h"
#include "mla.h"
#include "exe_graph/runtime/tiling_context.h"
#include "common.h"
#include "math.h"
#include "tiling/platform/platform_ascendc.h"

namespace AtbOps {

const int32_t NUM1 = 1;
const int32_t NUM2 = 2;
const int32_t NUM3 = 3;
const int32_t NUM4 = 4;
const int32_t NUM5 = 5;
const int32_t NUM16 = 16;
const int32_t NUM32 = 32;
const int32_t NUM64 = 64;
const int32_t NUM256 = 256;
const int32_t NUM512 = 512;
const int32_t NUM576 = 576;
const float SPLITKV_RATION = 0.8;


ge::graphStatus GetMLANdInfo(gert::TilingContext *context, MLAInfo &mmInfo,
                    OpParam::MLA &param)
{
    gert::Shape kcacheShape = context->GetInputTensor(DIM_2)->GetOriginShape();
    auto KDims = kcacheShape.GetDimNum();
    gert::Shape tableShape = context->GetInputTensor(DIM_4)->GetOriginShape();
    mmInfo.kNz = (kcacheShape.GetDim(KDims - 1) == NUM16 || kcacheShape.GetDim(KDims - 1) == NUM32) ? 1 : 0;
    if (mmInfo.kNz) {
        mmInfo.embeddingSize = static_cast<int32_t>(kcacheShape.GetDim(DIM_3)) *
                            static_cast<int32_t>(kcacheShape.GetDim(DIM_1));
        mmInfo.blockSize = static_cast<int32_t>(kcacheShape.GetDim(DIM_2));
    } else {
        mmInfo.embeddingSize = static_cast<int32_t>(kcacheShape.GetDim(DIM_3));
        mmInfo.blockSize = static_cast<int32_t>(kcacheShape.GetDim(DIM_1));
    }
    mmInfo.numTokens = static_cast<int32_t>(param.kvSeqLen.size());
    mmInfo.numBlocks = static_cast<int32_t>(kcacheShape.GetDim(DIM_0));
    mmInfo.maxNumBlocksPerQuery = static_cast<int32_t>(tableShape.GetDim(DIM_1));
    mmInfo.tor = param.tor;
    if (param.kvSeqLen.size() > 0) {
        mmInfo.kvSeqLen = param.kvSeqLen.data();
    }
    if (param.qSeqLen.size() > 0) {
        mmInfo.qSeqLen = param.qSeqLen.data();
    }
    param.kvHead = param.kvHead <= 0 ? param.headSize : param.kvHead;
    mmInfo.batch = static_cast<int32_t>(param.kvSeqLen.size());
    mmInfo.kvHeads = param.kvHead;
    mmInfo.numHeads = static_cast<int32_t>(param.headSize);
    mmInfo.maskType = static_cast<int32_t>(param.maskType);
    mmInfo.mtpTp1Flag = (mmInfo.numHeads == M_LIMIT) && (static_cast<int32_t>(mmInfo.type) < NUM2); // quant not support
    if (mmInfo.mtpTp1Flag) {
        mmInfo.maskType = 0;
    }
    if (static_cast<int32_t>(mmInfo.type) >= NUM2) {
        mmInfo.maskType = 0;
    }

    if (mmInfo.qSeqLen != nullptr) {
        mmInfo.totalTaskNum = std::accumulate(mmInfo.qSeqLen, mmInfo.qSeqLen + mmInfo.batch, static_cast<int32_t>(0));
    } else {
            mmInfo.totalTaskNum = mmInfo.batch;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetMLAInfo(gert::TilingContext *context, MLAInfo &mmInfo, OpParam::MLA &param)
{
    GetMLANdInfo(context, mmInfo, param);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetTilingKeyTypeBase(MLAInfo &mmInfo, const gert::Tensor *qTensor, const gert::Tensor *qRopeTensor)
{
    if (qTensor->GetDataType() == ge::DataType::DT_BF16) {
        mmInfo.type = TilingKeyType::TILING_BF16_DATA;
    } else if (qTensor->GetDataType() == ge::DataType::DT_FLOAT16) {
        mmInfo.type = TilingKeyType::TILING_HALF_DATA;
    } else if (qRopeTensor->GetDataType() == ge::DataType::DT_FLOAT16) {
        mmInfo.type = TilingKeyType::TILING_INT8_HALF_DATA;
    } else {
        mmInfo.type = TilingKeyType::TILING_INT8_BF16_DATA;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenTilingKey(gert::TilingContext *context, MLAInfo &mmInfo, OpParam::MLA &param)
{
    uint32_t dataType = static_cast<int32_t>(mmInfo.type);
    uint32_t tilingKey = dataType + (mmInfo.kNz << NUM4) + (mmInfo.mtpTp1Flag << NUM2) + (param.isRing << NUM5);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

OpParam::MLA GetParamFromTilingContext(gert::TilingContext *context) {
    OpParam::MLA param;
    param.type = static_cast<AtbOps::OpParam::MLA::Type>(*(context->GetAttrs()->GetInt(0)));
    param.headSize = static_cast<int32_t>(*(context->GetAttrs()->GetInt(1)));
    param.tor = *(context->GetAttrs()->GetFloat(2));
    param.kvHead = static_cast<int32_t>(*(context->GetAttrs()->GetInt(3)));
    param.maskType = static_cast<AtbOps::OpParam::MLA::MaskType>(*(context->GetAttrs()->GetInt(4)));
    param.isRing = static_cast<int32_t>(*(context->GetAttrs()->GetInt(7)));
    auto qSeqLen = context->GetAttrs()->GetListInt(5)->GetData();
    size_t arraySize = context->GetAttrs()->GetListInt(5)->GetSize();
    param.qSeqLen.reserve(arraySize);
    if (arraySize >= 1 && reinterpret_cast<const int32_t *>(qSeqLen)[0] >= 0) {
        for (size_t i = 0; i < arraySize; ++i) {
            param.qSeqLen.push_back(reinterpret_cast<const int32_t *>(qSeqLen)[i]);
        }
    }

    auto kvSeqLenAttr = context->GetAttrs()->GetListInt(6)->GetData();
    arraySize = context->GetAttrs()->GetListInt(6)->GetSize();
    param.kvSeqLen.reserve(arraySize);
    if (arraySize >= 1 && reinterpret_cast<const int32_t *>(kvSeqLenAttr)[0] >= 0) {
        for (size_t i = 0; i < arraySize; ++i) {
            param.kvSeqLen.push_back(reinterpret_cast<const int32_t *>(kvSeqLenAttr)[i]);
        }
    }
    return param;
}

uint64_t GetTilingSize(OpParam::MLA param) {
    int32_t TILING_PARA_SIZE = 8;
    int32_t TILING_HEAD_SIZE = 15;
    int32_t TILING_PARA_SIZE_TP1 = 4;
    uint32_t TILINGMIN = 512;
    int32_t M_LIMIT = 128;

    uint32_t launchBufferSize_ = AtbOps::Utils::RoundUp((TILING_PARA_SIZE + TILING_HEAD_SIZE) * sizeof(uint32_t), TILINGMIN);

    auto batch = param.kvSeqLen.size();
    if (param.headSize == M_LIMIT) {
        uint64_t taskNum = param.qSeqLen.data() == nullptr ? batch :
                           std::accumulate(param.qSeqLen.data(),
                                           param.qSeqLen.data() + batch, static_cast<int32_t>(0));
        uint64_t bufferSize =
                AtbOps::Utils::RoundUp(launchBufferSize_ + TILING_PARA_SIZE_TP1 * (taskNum - 1) * sizeof(uint32_t), TILINGMIN);
        return bufferSize;
    }

    uint64_t bufferSize =
            AtbOps::Utils::RoundUp(launchBufferSize_ + TILING_PARA_SIZE * (batch - 1) * sizeof(uint32_t), TILINGMIN);
    return bufferSize;
}

ge::graphStatus MLATiling(gert::TilingContext *context)
{
    OpParam::MLA param = GetParamFromTilingContext(context);
    auto qTensor = context->GetInputTensor(DIM_0);
    auto qRopeTensor = context->GetInputTensor(DIM_1);
    
    MLAInfo mmInfo = {};
    GetTilingKeyTypeBase(mmInfo, qTensor, qRopeTensor);
    GetMLAInfo(context, mmInfo, param);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    
    auto blockDim = ascendcPlatform.GetCoreNumAic();

    uint64_t tilingSizeWithoutWorkspaceParam = GetTilingSize(param);
    uint64_t tilingSizeWithWorkSpace = sizeof(uint64_t) * 6 + tilingSizeWithoutWorkspaceParam;
    uint32_t *tilingParam = static_cast<uint32_t *>(context->GetRawTilingData()->GetData());
    context->GetRawTilingData()->SetDataSize(tilingSizeWithWorkSpace);    
    
    ge::graphStatus ret = GetMLATilingParam(param, mmInfo, blockDim, tilingParam + 6*2, tilingSizeWithWorkSpace);
    if (ret != ge::GRAPH_SUCCESS) {
        printf("GetMLATilingParam failed: %d\n", ret);
        return ret;
    }

    uint32_t dataLenHalf = sizeof(uint16_t);
    uint32_t dataLenFloat = sizeof(float);
    uint32_t dataLenInt = sizeof(int32_t);
    uint64_t basicWorkSpaceHalf = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenHalf;
    uint64_t basicWorkSpaceFloat = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenFloat;
    uint64_t basicWorkSpaceInt8 = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenInt;
    bool isQuant = (static_cast<int32_t>(mmInfo.type) < NUM2) ? 0 : 1;
    uint64_t pWorkSpaceSize = isQuant ? basicWorkSpaceInt8 : basicWorkSpaceHalf * 2;
    uint64_t oTempWorkSpcaceSize = isQuant ? basicWorkSpaceInt8 * 2 : basicWorkSpaceFloat * 2;
    uint64_t tailWorkSpaceFloat = blockDim * 128 * 2 * dataLenFloat;
    uint64_t *workspaceParam = reinterpret_cast<uint64_t *>(tilingParam);
    if (isQuant) {
        workspaceParam[0] = basicWorkSpaceFloat;
        workspaceParam[1] = basicWorkSpaceFloat;
        workspaceParam[2] = pWorkSpaceSize;
        workspaceParam[3] = oTempWorkSpcaceSize;
        workspaceParam[4] = basicWorkSpaceFloat;
        workspaceParam[5] = tailWorkSpaceFloat;
    } else {
        workspaceParam[0] = basicWorkSpaceFloat * 2;
        workspaceParam[1] = NUM512;
        workspaceParam[2] = pWorkSpaceSize;
        workspaceParam[3] = oTempWorkSpcaceSize;
        workspaceParam[4] = basicWorkSpaceFloat;
        workspaceParam[5] = tailWorkSpaceFloat;
    }

    uint64_t usrSize =
            workspaceParam[0] + workspaceParam[1] + workspaceParam[2] + workspaceParam[3] + workspaceParam[4] + workspaceParam[5];
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;

    ge::graphStatus ret2 = GenTilingKey(context, mmInfo, param);
    if (ret2 != ge::GRAPH_SUCCESS) {
        printf("GetMLATilingParam failed: %d\n", ret2);
        return ret2;
    }

    context->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

} // namespace AtbOps
