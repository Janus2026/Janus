/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mal_preprocess_tilingdata.h
 * \brief
 */

#ifndef OPTILING_MLA_PREPROCESS_TILING_DATA
#define OPTILING_MLA_PREPROCESS_TILING_DATA

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MlaPpMatmulTilingData)
TILING_DATA_FIELD_DEF(int64_t, numBatch);
TILING_DATA_FIELD_DEF(int64_t, m);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, m0);
TILING_DATA_FIELD_DEF(int64_t, k0);
TILING_DATA_FIELD_DEF(int64_t, n0);
TILING_DATA_FIELD_DEF(int64_t, mLoop);
TILING_DATA_FIELD_DEF(int64_t, kLoop);
TILING_DATA_FIELD_DEF(int64_t, nLoop);
TILING_DATA_FIELD_DEF(int64_t, coreLoop);
TILING_DATA_FIELD_DEF(int64_t, swizzleCount);
TILING_DATA_FIELD_DEF(int64_t, swizzleDirect);
TILING_DATA_FIELD_DEF(int64_t, enShuffleK);
TILING_DATA_FIELD_DEF(int64_t, blockDim);
TILING_DATA_FIELD_DEF(int64_t, enLoadAllAmat);
TILING_DATA_FIELD_DEF(int64_t, b0matPingPongBufferLen);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MlaPpMatmulTilingDataOp, MlaPpMatmulTilingData)

BEGIN_TILING_DATA_DEF(MlaTilingData)
TILING_DATA_FIELD_DEF(int64_t, numCore);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, perTaskNum);
TILING_DATA_FIELD_DEF(int64_t, resTaskNum);
TILING_DATA_FIELD_DEF_STRUCT(MlaPpMatmulTilingData, mm1);
TILING_DATA_FIELD_DEF_STRUCT(MlaPpMatmulTilingData, mm2);
TILING_DATA_FIELD_DEF_STRUCT(MlaPpMatmulTilingData, mm3);
// rms1
TILING_DATA_FIELD_DEF(int64_t, rmsNumCore1);
TILING_DATA_FIELD_DEF(int64_t, rmsNumCol1);
TILING_DATA_FIELD_DEF(int64_t, rmsNumRow1);
TILING_DATA_FIELD_DEF(int64_t, rmsQuantMin1);
TILING_DATA_FIELD_DEF(int64_t, hiddtenState);
// rms2
TILING_DATA_FIELD_DEF(int64_t, rmsNumCore2);
TILING_DATA_FIELD_DEF(int64_t, rmsNumCol2);
TILING_DATA_FIELD_DEF(int64_t, rmsNumRow2);
TILING_DATA_FIELD_DEF(int64_t, rmsQuantMin2);

TILING_DATA_FIELD_DEF(int64_t, hiddenSizeQ);
TILING_DATA_FIELD_DEF(int64_t, headNumQ);
TILING_DATA_FIELD_DEF(int64_t, headDim);
TILING_DATA_FIELD_DEF(int64_t, concatSize);
TILING_DATA_FIELD_DEF(int64_t, rotaryCoeff);
TILING_DATA_FIELD_DEF(int64_t, ntokens);
TILING_DATA_FIELD_DEF(int64_t, realCore);
TILING_DATA_FIELD_DEF(int64_t, nlCoreRun);
TILING_DATA_FIELD_DEF(int64_t, lCoreRun);
TILING_DATA_FIELD_DEF(int64_t, maxNPerLoopForUb);
TILING_DATA_FIELD_DEF(int64_t, preCoreLoopTime);
TILING_DATA_FIELD_DEF(int64_t, preCoreLoopNLast);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLoopTime);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLoopNLast);
// EinSumQuant
TILING_DATA_FIELD_DEF(int64_t, esqFrontCore);
TILING_DATA_FIELD_DEF(int64_t, esqTailCore);
TILING_DATA_FIELD_DEF(int64_t, esqFrontCoreBatch);
TILING_DATA_FIELD_DEF(int64_t, esqTailCoreBatch);
TILING_DATA_FIELD_DEF(int64_t, esqHeadNum);
TILING_DATA_FIELD_DEF(int64_t, esqColNum);
TILING_DATA_FIELD_DEF(int64_t, esqUbHeadLoop);
TILING_DATA_FIELD_DEF(int64_t, esqHeadPerLoop);
TILING_DATA_FIELD_DEF(int64_t, esqHeadTail);
TILING_DATA_FIELD_DEF(int64_t, esqColLoop);
TILING_DATA_FIELD_DEF(int64_t, esqColTail);
// workspace的参数配置
TILING_DATA_FIELD_DEF(int64_t, maxWorkspaceSize);
TILING_DATA_FIELD_DEF(float, epsilon);
// others
TILING_DATA_FIELD_DEF(bool, doRmsNorm);
TILING_DATA_FIELD_DEF(bool, qDownOutFlag);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MlaPreprocess, MlaTilingData)
REGISTER_TILING_DATA_CLASS(MlaPreprocessV2, MlaTilingData)

} // namespace optiling

#endif // OPTILING_MLA_PREPROCESS_TILING_DATA