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
#ifndef ATB_SPEED_MODELS_QWEN_DECODER_LAYER_H
#define ATB_SPEED_MODELS_QWEN_DECODER_LAYER_H

#include <vector>
#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace qwen {
struct DecoderLayerParam {
    bool isFA = false;
    bool isPrefill = false;
    bool isBF16 = false;
    bool isPack = true;
    bool supportSwiGLU = false;
    bool supportLcoc = false;
    bool supportSpeculate = false;
    bool enableSplitFuse = false;
    bool supportLora = false;
    bool loraEnableGMM = false;
    bool enableLogN = false;
    bool kvQuant = false;
    bool enableIntraLayerAddNorm= false;
    bool enableInterLayerAddNorm= false;
    std::string backend = "hccl";
    int rank = 0;
    int worldSize = 1;
    int quantType = 0;
    int quantGroupSize = 64;
    int numAttentionHeadsPerRank = 0;
    int hiddenSizePerAttentionHead = 0;
    int numKeyValueHeadsPerRank = 0;
    float rmsNormEps = 0;

    std::vector<int> seqLen;
    std::vector<int> tokenOffset;
    std::vector<int> packQuantType = {};  // translated,translatedQKV packtranslated,translatedMLP packtranslated
    // translated,translatedq,k,v,self attention out,gate,up,down lineartranslated
    std::vector<int> linearQuantType = {};
    std::vector<int> linearTransposeType;
};


atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation);
}  // namespace qwen
}  // namespace atb_speed
#endif