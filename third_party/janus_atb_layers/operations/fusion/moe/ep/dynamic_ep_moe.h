/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#ifndef ATB_SPEED_MODELS_DYNAMIC_EP_MOE_OPERATION_H
#define ATB_SPEED_MODELS_DYNAMIC_EP_MOE_OPERATION_H
#include <atb/atb_infer.h>
#include <atb/comm.h>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/log.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {
struct DynamicEpMoEParam {
    bool transpose = true;
    bool supportSwiGLU = true;
    int32_t topk = 2;
    int32_t scaledTopk = -1; /// translateddeepseektranslatedscaledTopktranslated
    bool enableInitRoutingCutoff = false;  /// A flag indicating whether to use scaled topk option
    int gmmQuantType = 0;
    uint32_t numOfExperts = 8;
    uint32_t numOfDeviceExperts = 8;
    std::vector<int> moeLinearQuantType = {};
    bool hasBias = false;
    bool isBF16 = false;
    bool gateUpTransposeB = false;
    bool downTransposeB = false;
    bool isDynamicEp = false;
    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    int quantGroupSize = 0; /// Group size of per-group quantization

    std::vector<int32_t> deviceExpert = {0, 1, 2, 3, 4, 5, 6, 7};
    int expertParallelDegree = 0;
    bool enableFusedRouting = false;
    bool enableGMMSwigluQuant = false;
    bool enableInitQuant = false;
    bool enableSwigluQuant = false;
    bool enableAtlasGMMFused = false;
    bool enableFusedTopk = false;
    bool enableDispatchCombineV2 = false; /// A flag indicating whether to use dispatch_v2 and combine_v2
    atb_speed::common::OpBackend swigluBackend = atb_speed::common::OpBackend::ATB;
    std::string backend = "hccl";
    HcclComm hcclComm = nullptr;

    bool hasMoeEp = false;
    int moeEpRank = 0;
    int moeEpSize = 1;
    int maxDecodeDpTokenSize = 0;
    std::string moeEpDomain = "";
    std::string moeEpRankTableFile = "";

    bool hasMlpTp = false;
    int mlpTpRank = 0;
    int mlpTpSize = 1;
    std::string mlpTpDomain = "";
    std::string mlpTpRankTableFile = "";
    bool enableCVOverlap = false; /// A flag indicating whether the model use cube and vector parallel
    bool enableMoeDistribute = false;
    bool enableExpertCumSumOutput = false;
    bool enableGatingDp = false;
    int64_t numDanglingSharedExperts = 0;
    uint32_t numOfRedundantExpert = 0;
    bool enableIndexGmm = false;
    bool enableLcocAll2All = false;
    std::string routingMethod = "";

    std::string lcclMoeEpDomain = "";
    HcclComm lcclMoeEpHcclComm = nullptr;

    bool mixSharedRouting = false;
    bool enableInitRoutingV3 = false;
};

atb::Status CreateDynamicEpMoEOperation(const DynamicEpMoEParam &param, atb::Operation **operation);
}
} // namespace atb_speed
#endif
