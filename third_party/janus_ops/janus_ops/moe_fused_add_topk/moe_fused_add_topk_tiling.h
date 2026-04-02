/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct MoeFusedAddTopkCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
};

BEGIN_TILING_DATA_DEF(MoeFusedAddTopkTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, firstDimSize);
    TILING_DATA_FIELD_DEF(uint32_t, secondDimSize);
    TILING_DATA_FIELD_DEF(uint32_t, addNumDimSize);
    TILING_DATA_FIELD_DEF(uint32_t, groupNum);
    TILING_DATA_FIELD_DEF(uint32_t, groupTopk);
    TILING_DATA_FIELD_DEF(uint32_t, topN);
    TILING_DATA_FIELD_DEF(uint32_t, topK);

    TILING_DATA_FIELD_DEF(uint32_t, activateType);
    TILING_DATA_FIELD_DEF(uint32_t, isNorm);
    TILING_DATA_FIELD_DEF(float, scale);
    TILING_DATA_FIELD_DEF(uint32_t, groupEles);
    TILING_DATA_FIELD_DEF(uint32_t, blockNum);
    TILING_DATA_FIELD_DEF(uint32_t, ubFactorElement);
    TILING_DATA_FIELD_DEF(uint32_t, batchPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tailBatch);

    TILING_DATA_FIELD_DEF(uint32_t, expertNum);
    TILING_DATA_FIELD_DEF(uint32_t, tableDim);
    TILING_DATA_FIELD_DEF(uint32_t, topkMaxValue);
    TILING_DATA_FIELD_DEF(uint32_t, topkMinValue);
    TILING_DATA_FIELD_DEF(uint32_t, reserved);
    TILING_DATA_FIELD_DEF(uint64_t, workspacePerCore);
    TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeFusedAddTopk, MoeFusedAddTopkTilingData);
} // namespace optiling

//#endif // ASCEND_OPS_MOE_FUSED_ADD_TOPK_TILING_H
