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
 * \file mla_preprocess_tiling.h
 * \brief
 */
#ifndef OPTILING_PARAMS_PREPROCESS_TILING
#define OPTILING_PARAMS_PREPROCESS_TILING

#include <cstdint>
#include <string>
#include <sstream>
#include "mla_preprocess_tilingdata.h"
#include "../../common/tiling/tiling_base.h"
#include "register/op_impl_registry.h"

namespace optiling {
namespace OpParam {
struct MlaPreprocessParam {
    enum class QuantMode : uint64_t {
        PER_TENSOR_ASYMM_QUANT = 0,
        PER_TOKEN_SYMM_QUANT,
        PER_TOKEN_ASYMM_QUANT,
        NO_QUANT,
    };
    uint64_t N = 128;
    uint64_t headNum = 0;
    uint64_t cacheMode = 0;
    QuantMode quantMode = QuantMode::PER_TENSOR_ASYMM_QUANT;
    bool operator==(const MlaPreprocessParam &other) const
    {
        return N == other.N && headNum == other.headNum && cacheMode == other.cacheMode && quantMode == other.quantMode;
    }
};
} // namespace OpParam

struct MlaPreProcessCompileInfo {};

class MlaPreprocessTiling {
public:
    optiling::MlaTilingData mlaTilingData;

    ge::graphStatus Init(gert::TilingContext *context);

    void RmsNormQuantTiling(const uint64_t numTokens, const uint64_t numVectorCore, const uint64_t hiddtenState);
    void RopeConcatTiling(const OpParam::MlaPreprocessParam &param, const uint64_t &aicNum);
    void EinSumQuantTiling(const OpParam::MlaPreprocessParam &param, const uint64_t &aicNum,
                           const ge::DataType inDtype, const bool doRmsQuant);
    void SetTilingKey(const ge::DataType inDtype, const OpParam::MlaPreprocessParam &param, const bool doRmsQuant,
                      gert::TilingContext *context);
    void SetMlapoWorkSpace(const ge::DataType inDtype, const OpParam::MlaPreprocessParam &param,
                           uint32_t sysWorkSpaceSize, gert::TilingContext *context);
    void PrintTilingData(gert::TilingContext *context);
    void PrintFirstTilingData(gert::TilingContext *context);
    void PrintLastTilingData(gert::TilingContext *context);
    OpParam::MlaPreprocessParam GetParam(gert::TilingContext *context);
};

} // namespace optiling

#endif // OPTILING_PARAMS_MLA_PRE_H
