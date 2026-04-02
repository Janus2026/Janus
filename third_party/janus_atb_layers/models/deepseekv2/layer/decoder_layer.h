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
#ifndef ATB_SPEED_MODELS_DEEPSEEK_V2_DECODER_LAYER_H
#define ATB_SPEED_MODELS_DEEPSEEK_V2_DECODER_LAYER_H

#include <vector>
#include <atb/comm.h>
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "atb_speed/base/external_comm_manager.h"
#include "atb_speed/utils/operation_util.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace deepseekV2 {
class DecoderLayerParam : public atb_speed::moe::MoeLayerParam {
public:
    bool enableFusedRouting = true;
    bool hasSharedExpert = true;
    bool hasSharedExpertGate = false;
    bool isDenseLayer = false;
    bool isLastLayer = false;
    bool isDynamicEp = false;
    bool hasP2DWeight = false;
    bool enableCVOverlap = false; /// A flag indicating whether the model use cube and vector parallel
    bool enableExtraOprojTp = false;
    bool enableATBGateMatmul = false;
    bool enableMlaPrefetch = false;
    int maskStartIdx = 0;
    int layerId = 0;
    int numHiddenLayers = 0;
    int firstKDenseReplace = 1;
    int numOfSharedExperts = 2;       // 2:Defaulting the number of shared experts to 2
    int rank = 0;
    int worldSize = 1;
    // quant translated
    int mlpNormQuantType = atb::infer::QUANT_UNDEFINED;
    bool isAntiOutlier = false;
    // Grouped topktranslated
    int numOfGroups = 1;
    int scaledTopk = -1; /// translateddeepseektranslatedscaledTopktranslated
    bool enableInitRoutingCutoff = false;  /// A flag indicating whether to use scaled topk option
    float routedScalingFactor = 1;
    bool enableFusedTopk = false;
    // MLAtranslated
    int qLoraRank = 1536;
    int kvLoraRank = 512;
    int headNum = 128;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    float softmaxScale = 0;
    bool enableMlaPreprocess = false;
    bool enableCustomizeMla = false;
    bool isNzCache = false;
    bool enablePrefixCache = false;
    // lightindexertranslated
    int index_head_dim = 0; // 128
    int index_n_heads = 0; // 64
    int index_topk = 0; // 2048
    // translated
    int attnStreamNum = 1;
    int ffnStreamNum = 1;
    int lmheadStreamNum = 1;
    bool attnAllreduce = false;
    bool attnReduceScatter = false;
    bool attnAllGather = false;
    bool ffnAllreduce = false;
    bool ffnReduceScatter = false;
    bool ffnAllGather = false;
    bool hasAttnComm = false;
    bool hasFfnComm = false;
    bool enableExpertCumSumOutput = false;
    bool isMlpFullTP = false;
    std::string routingMethod = "deviceLimited";
    std::string processLogits = "scaling";
    std::string backend = "hccl";
    std::string rankTableFile = "";
    std::vector<int> attnLinearQuantType = {};
    std::vector<int> attnLinearTransposeType = {};
    atb::SVector<int32_t> topkGroups = {1}; // num of selected groups
    int moePackQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    // h3p
    bool enableQkvdownDp = false;
    bool enableSharedExpertDp = false;
    bool enableGatingDp = false;
    bool enableSharedExpertOverlap = false;

    bool enableLoadBalance = false;
    bool maskfree = true;

    bool enableAllToAllMC2 = false;
    HcclComm hcclComm = nullptr;
    bool enableGatherPreNorm = false;
    bool enableEPWB = false;
    uint32_t numOfRedundantExpert = 0;
    int64_t numDanglingSharedExperts = 0;

    HcclComm dispatchAndCombineHcclComm;
    std::string dispatchAndCombinecommDomain = "";

    bool enableInfNan = true;
    bool enableOutLcocTp = false;
    bool enablePreprocessLcocTp = false;
    bool enableLcocAll2All = false;
    bool mixSharedRouting = false;
    bool enableFusedMLA = false;
    bool enableIndexGmm = false;
};

/// The index of the GATEUP linear within the mlp
const uint64_t MLP_GATEUP_LINEAR_INDEX = 0;
/// The index of the down linear within the mlp
const uint64_t MLP_DOWN_LINEAR_INDEX = 2;
/// The index of the GATEUP linear within the moe
const uint64_t MOE_GATEUP_LINEAR_INDEX = 1;
/// The index of the down linear within the moe
const uint64_t MOE_DOWN_LINEAR_INDEX = 3;

atb::Status DecoderLayer(DecoderLayerParam &param, atb::Operation **operation);

class DecoderLayer {
public:
    explicit DecoderLayer();
    ~DecoderLayer();

private:
    int32_t layerId_ = 0;
};

}  // namespace deepseekV2
}  // namespace atb_speed
#endif
