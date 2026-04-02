/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __X_FLASH_ATTENTION_INFER_TILINGDATA_H__
#define __X_FLASH_ATTENTION_INFER_TILINGDATA_H__

#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {

enum InputPosIndex {
    QUERY = 0,
    KEY,
    VALUE,
    MASK,
    BLOCK_TABLE,
    Q_LENS,
    KV_LENS,
};

enum AttrsIndex {
    LAYOUT_IDX = 0,
    QHEAD_IDX,
    KVHEAD_IDX,
    SCALE_IDX,
};

struct FlashAttentionScoreCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    platform_ascendc::SocVersion socVersion;
};

constexpr int32_t NUM2 = 2;
constexpr int32_t NUM3 = 3;
constexpr int32_t NUM4 = 4;
constexpr int32_t NUM5 = 5;
constexpr uint32_t Q_TILE_CEIL = 128;
constexpr uint32_t NZ_LAST_DIM = 16;
constexpr uint32_t MAX_KV_STACK_LEN = 1024;
const uint32_t SIZE_OF_32BIT = 4;
constexpr int32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
uint32_t GetQNBlockTile(int64_t qSeqlen, uint32_t groupSize)
{
    uint32_t qRowNumCeil = Q_TILE_CEIL;
    uint32_t qNBlockTile = (qRowNumCeil / qSeqlen) / 2 * 2;
    qNBlockTile = std::min(qNBlockTile, groupSize);
    qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
    return qNBlockTile;
}

uint32_t GetQSBlockTile(int64_t kvSeqlen)
{
    uint32_t qSBlockTile = Q_TILE_CEIL;
    return qSBlockTile;
}

uint32_t GetKSBlockTile(int64_t kvSeqlen)
{
    uint32_t kSBlockTile = MAX_KV_STACK_LEN;
    return kSBlockTile;
}

BEGIN_TILING_DATA_DEF(XFAInferTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numTokens);
  TILING_DATA_FIELD_DEF(uint32_t, numHeads);
  TILING_DATA_FIELD_DEF(uint32_t, embeddingSize);
  TILING_DATA_FIELD_DEF(uint32_t, embeddingSizeV);
  TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
  TILING_DATA_FIELD_DEF(uint32_t, blockSize);
  TILING_DATA_FIELD_DEF(uint32_t, maxKvSeqlen);
  TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, maxNumBlocksPerBatch);
  TILING_DATA_FIELD_DEF(uint32_t, firstBatchTaskNum);
  TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
  TILING_DATA_FIELD_DEF(uint32_t, maskType);
  TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize);
  TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize);
  TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize);
  TILING_DATA_FIELD_DEF(uint64_t, updateSize);
  TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize);
  TILING_DATA_FIELD_DEF(float, scaleValue);
  TILING_DATA_FIELD_DEF(uint64_t, splitLseTotalSize);
  TILING_DATA_FIELD_DEF(uint64_t, splitOTotalSize);
  TILING_DATA_FIELD_DEF(uint32_t, totalSplitNodeNum);
  TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(XFlashAttentionInfer, XFAInferTilingData)


class XFAInferTiling {
  public:
    explicit XFAInferTiling(gert::TilingContext* tiling_context)
        : tiling_context_(tiling_context) {}
    ge::graphStatus RunTiling();
  private:
    uint64_t GetTilingKey() const;
    ge::graphStatus ParseInputShapeAndAttrs();
    ge::graphStatus FillBasicTilingData();
    void FillSplitCoreTilingDataForJD();
    void SetWorkspaces();
  private:
    XFAInferTilingData tiling_data_;
    gert::TilingContext* tiling_context_ = nullptr;
    uint32_t cubeCoreNum;
    uint32_t vecCoreNum;
    uint32_t inputDtype{0};
    uint32_t maskType{0};
    std::string kvLayout = "TND";
    std::string layout = "TND";
    bool pagedCacheFlag = true;
    uint32_t blockNum_;
    bool usingFD = true;
};

}  // namespace optiling
#endif  // __X_FLASH_ATTENTION_INFER_TILINGDATA_H__

