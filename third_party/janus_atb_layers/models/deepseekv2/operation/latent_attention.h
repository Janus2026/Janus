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

#ifndef ATB_SPEED_MODELS_COMMON_LATENT_ATTENTION_H
#define ATB_SPEED_MODELS_COMMON_LATENT_ATTENTION_H

#include <vector>
#include <map>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/embedding/positional_embedding.h"

namespace atb_speed {
namespace deepseekV2 {

constexpr uint64_t Q_PROJ_A_LINEAR_INDEX = 0;
constexpr uint64_t Q_PROJ_B_LINEAR_INDEX = 1;
constexpr uint64_t KV_PROJ_A_LINEAR_INDEX = 2;
constexpr uint64_t KV_PROJ_B_FOR_Q_LINEAR_INDEX = 3;
constexpr uint64_t KV_PROJ_B_FOR_V_LINEAR_INDEX = 4;
constexpr uint64_t O_LINEAR_INDEX = 5;

template <typename NormParamType>
struct LatentAttentionParam {
    // QKV linear param
    bool isGroupedQueryAttention = false;
    bool isBF16 = false;
    bool splitWithStride = false;
    bool qkvHasBias = false;
    bool skipNorm = false;
    bool normHasBias = false;
    bool enableNormQuantOp = true;
    int quantGroupSize = 0;
    // MLA Param
    int qLoraRank = 1536;
    int kvLoraRank = 512;
    int headNum = 128;
    int actual_headNum = 0;
    bool ropeTrans = false;

    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    bool enableMlaPreprocess = false;
    bool enableExtraOprojTp = false;
    bool enableCustomizeMla = false;
    bool isNzCache = false;
    // LightIndexer Param
    int index_head_dim = 0; // 128
    int index_n_heads = 0; // 64
    int index_topk = 0; // 2048

    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;
    std::vector<int> attnLinearQuantType = {};
    std::vector<int> attnLinearTransposeType = {};
    NormParamType normParamType;
    NormParamType normQuantParamType;
    // rope param
    atb_speed::common::RotaryType rotaryType;
    atb::infer::RopeParam ropeParam;
    // self attention param
    bool enableLogN = false;
    bool isFA = true;
    bool isPrefill = false;
    int headDim = 0;
    atb::infer::SelfAttentionParam selfAttentionParam;
    atb::infer::RingMLAParam ringMLAParam;
    atb::infer::PagedAttentionParam pageAttentionParam;
    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    // self out linear param
    bool selfAttnHasBias = false;
    bool enableLcoc = false;
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    atb_speed::common::TensorParallelInfo selfOutLinearTensorParallelInfo;
    atb_speed::common::ParallelInfo selfOutLinearInnerTensorParallelInfo;
    // context parallelism
    atb_speed::common::ParallelInfo contextParallelInfo;
    // sequence parallelism
    bool hasAttnInnerSp = false;
    int attnSpRank = 0;
    int attnSpSize = 1;
    std::string attnSpDomain = "";
    std::string attnSpRankTableFile = "";
    std::string attnSpBackend = "";
    HcclComm attnSpHcclComm = nullptr;
    bool attnOprojPrefetch = false;
    // h3p qkvdown dp
    int layerId = 0;
    int firstKDenseReplace = 0;
    bool isDenseLayer = false;
    bool enableQkvdownDp = false;
    bool hasAttnComm = false;
    bool hasFfnComm = false;
    int attnTpRank = 0;
    int attnTpSize = 1;
    std::string attnTpBackend = "";
    std::string attnTpDomain = "";
    std::string attnTpRankTableFile = "";
    HcclComm hcclComm = nullptr;
    bool ffnAllGather = false;

    bool enableOutLcocTp = false;
    bool enablePreprocessLcocTp = false;
    int lcocAttnTpRank = 0;
    int lcocAttnTpRankSize = 1;
    std::string lcocAttnTpBackend = "";
    std::string lcocAttnTpDomain = "";
    HcclComm lcocHcclComm = nullptr;
    bool enablePrefixCache = false;
    bool enableFusedMLA = false;
    float normEps = 0;
    float softmaxScale = 0;
};

// template <typename NormParamType>
// std::map<std::string, uint32_t> ConstructTensorMap(
//     uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);

template <typename NormParamType>
atb::Status Attention(const LatentAttentionParam<NormParamType> &param, atb::Operation **operation);
} // namespace deepseekV2
} // namespace atb_speed
#endif
