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

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(XAttentionTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numHeads);
  TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
  TILING_DATA_FIELD_DEF(uint32_t, embeddingSize);
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, beamSize);
  TILING_DATA_FIELD_DEF(float, scaleValue);
  TILING_DATA_FIELD_DEF(uint32_t, maskType);
  TILING_DATA_FIELD_DEF(uint32_t, numTokens);
  TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
  TILING_DATA_FIELD_DEF(uint32_t, blockSize);
  TILING_DATA_FIELD_DEF(uint32_t, sharedCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, maxNumBlocksPerBatch);
  TILING_DATA_FIELD_DEF(uint32_t, firstSharedBatchTaskNum);
  TILING_DATA_FIELD_DEF(uint32_t, sharedTotalTaskNum);
  TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize);
  TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize);
  TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize);
  TILING_DATA_FIELD_DEF(uint64_t, updateSize);
  TILING_DATA_FIELD_DEF(uint32_t, rowSumMaxSize);
  TILING_DATA_FIELD_DEF(uint64_t, sharedWorkspaceSize);
  TILING_DATA_FIELD_DEF(uint32_t, groupSize);
  TILING_DATA_FIELD_DEF(uint32_t, maxDecodeStep);
  TILING_DATA_FIELD_DEF(uint32_t, unsharedCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, unshareGroupCountPerLoop);
  TILING_DATA_FIELD_DEF(uint32_t, unsharedFullCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, unsharedTaskNumHead);
  TILING_DATA_FIELD_DEF(uint32_t, unsharedTaskNumTail);
  TILING_DATA_FIELD_DEF(uint32_t, unsharedLoopCountPerBatch);
  TILING_DATA_FIELD_DEF(uint32_t, combineFormerCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, combineFormerRowNum);
  TILING_DATA_FIELD_DEF(uint32_t, combineTailRowNum);
  TILING_DATA_FIELD_DEF(uint32_t, combineCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(XAttention, XAttentionTilingData)
}
