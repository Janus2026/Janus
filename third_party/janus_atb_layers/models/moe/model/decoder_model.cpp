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
#include <set>
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "models/moe/model/decoder_model.h"

#include <atb/types.h>

namespace atb_speed {
namespace moe {

constexpr size_t MOE_LINEAR_TYPE_LENGTH = 4;

HcclComm MoeModelParam::dispatchAndCombineHcclComm = nullptr;
std::string MoeModelParam::dispatchAndCombinecommDomain = "";

void MoeModelParam::SetHcclCommForDispatchAndCombine() const
{
    if (!isPrefill && enableAllToAllMC2 && expertParallelDegree == 2) { // 2: dynamic ep level
        // Assign commDomain by rankIds and rank
        if (dispatchAndCombineHcclComm != nullptr) {
            ATB_SPEED_LOG_DEBUG("Reuse the hccl communication group for dispatch and combine.");
        } else {
            atb_speed::common::ParallelInfo moeEpParallelInfo = mapping.Get(base::MOE_EP);
            dispatchAndCombinecommDomain = GetSingleton<ExternalCommManager>().GetCommDomain(
                moeEpParallelInfo.groupId, moeEpParallelInfo.rankIds, moeEpParallelInfo.rank,
                moeEpParallelInfo.defaultBackend, moeEpParallelInfo.bufferSize, 0, false);  // 0: Default Stream Id

            dispatchAndCombineHcclComm = \
                GetSingleton<ExternalCommManager>().GetCommPtr(dispatchAndCombinecommDomain);
            ATB_SPEED_LOG_DEBUG("Create the hccl communication group for dispatch and combine.");
        }
    }
}

void MoeModelParam::ParseParam(const nlohmann::json &paramJson)
{
    atb_speed::base::ModelParam::ParseParam(paramJson);
    ParseQuantParams(paramJson);
    ParseAttnParallelParams(paramJson);
    ParseParallelParams(paramJson);
    ParseInteOpParams(paramJson);
    ParseMoEParams(paramJson);
    CheckParallelParamValid();
    if (paramJson.contains("numOfExperts")) {
        numOfExperts = atb_speed::base::FetchJsonParam<uint32_t>(paramJson, "numOfExperts");
    }
    if (paramJson.contains("numOfDeviceExperts")) {
        numOfDeviceExperts = atb_speed::base::FetchJsonParam<uint32_t>(paramJson, "numOfDeviceExperts");
    }
    if (paramJson.contains("expertParallelDegree")) {
        this->expertParallelDegree = paramJson["expertParallelDegree"].get<int>();
    }
    if (paramJson.contains("routingMethod")) {
        this->routingMethod = paramJson["routingMethod"].get<std::string>();
    }
    if (paramJson.contains("processLogits")) {
        this->processLogits = paramJson["processLogits"].get<std::string>();
    }
    if (paramJson.contains("normHasBias")) {
        this->normHasBias = paramJson["normHasBias"].get<bool>();
    }
    if (paramJson.contains("firstKDenseReplace")) {
        this->firstKDenseReplace = atb_speed::base::FetchJsonParam<int>(paramJson, "firstKDenseReplace");
    }
    if (paramJson.contains("numOfSharedExperts")) {
        this->numOfSharedExperts = atb_speed::base::FetchJsonParam<int>(paramJson, "numOfSharedExperts");
    }
    if (paramJson.contains("hasSharedExpert")) {
        this->hasSharedExpert = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasSharedExpert");
    }
    if (paramJson.contains("hasSharedExpertGate")) {
        this->hasSharedExpertGate = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasSharedExpertGate");
    }
    if (paramJson.contains("maskStartIdx")) {
        maskStartIdx = atb_speed::base::FetchJsonParam<int>(paramJson, "maskStartIdx");
    }
    if (paramJson.contains("numOfSelectedExperts")) {
        for (auto item : paramJson["numOfSelectedExperts"]) {
            this->numOfSelectedExperts.push_back(item.get<int>());
        }
    }
    if (paramJson.contains("numOfGroups")) {
        this->numOfGroups = atb_speed::base::FetchJsonParam<int>(paramJson, "numOfGroups");
    }
    if (paramJson.contains("numOfSelectedGroups")) {
        this->topkGroups.push_back(atb_speed::base::FetchJsonParam<int>(paramJson, "numOfSelectedGroups"));
    }
    if (paramJson.contains("deviceExpert")) {
        for (auto item : paramJson["deviceExpert"]) {
            deviceExpert.push_back(atb_speed::base::FetchJsonParam<int32_t>(item, "deviceExpert", true));
        }
    }
    if (paramJson.contains("enableGMMSwigluQuant")) {
        this->enableGMMSwigluQuant = paramJson["enableGMMSwigluQuant"].get<bool>();
    }
    if (paramJson.contains("enableAtlasGMMFused")) {
        this->enableAtlasGMMFused = paramJson["enableAtlasGMMFused"].get<bool>();
    }
    if (paramJson.contains("enableDpOut")) {
        this->enableDpOut = paramJson["enableDpOut"].get<bool>();
    }
    if (paramJson.contains("lmHeadLocalTp")) {
        this->lmHeadLocalTp = paramJson["lmHeadLocalTp"].get<bool>();
    }
    if (paramJson.contains("enableDispatchCombineV2")) {
        this->enableDispatchCombineV2 = paramJson["enableDispatchCombineV2"].get<bool>();
    }
    SetHcclCommForDispatchAndCombine();
}

void MoeModelParam::ParseInteOpParams(const nlohmann::json &paramJson)
{
    if (paramJson.contains("enableFusedRouting")) {
        this->enableFusedRouting = paramJson["enableFusedRouting"].get<bool>();
    }
    if (paramJson.contains("enableInitQuant")) {
        this->enableInitQuant = paramJson["enableInitQuant"].get<bool>();
    }
    if (paramJson.contains("enableSwigluQuant")) {
        this->enableSwigluQuant = paramJson["enableSwigluQuant"].get<bool>();
    }
}

void MoeModelParam::ParseQuantParams(const nlohmann::json &paramJson)
{
    for (auto item : paramJson["moeLinearQuantType"]) {
        this->moeLinearQuantType.push_back(item.get<std::vector<int>>());
    }
    for (auto item : paramJson["mlpLinearQuantType"]) {
        this->mlpLinearQuantType.push_back(item.get<std::vector<int>>());
    }
    for (auto item : paramJson["moeLinearTransposeType"]) {
        this->moeLinearTransposeType.push_back(item.get<std::vector<int>>());
    }
    for (auto item : paramJson["mlpLinearTransposeType"]) {
        this->mlpLinearTransposeType.push_back(item.get<std::vector<int>>());
    }
}

void MoeModelParam::ParseAttnParallelParams(const nlohmann::json &paramJson)
{
    if (paramJson.contains("hasAttnTp")) {
        hasAttnTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasAttnTp");
    }
    if (paramJson.contains("attnTpRank")) {
        attnTpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "attnTpRank");
    }
    if (paramJson.contains("attnTpSize")) {
        attnTpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "attnTpSize"));
    }
    if (paramJson.contains("attnTpDomain")) {
        attnTpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "attnTpDomain");
    }
    if (paramJson.contains("hasAttnOprojTp")) {
        hasAttnOprojTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasAttnOprojTp");
    }
    if (paramJson.contains("attnOprojTpRank")) {
        attnOprojTpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "attnOprojTpRank");
    }
    if (paramJson.contains("attnOprojTpSize")) {
        attnOprojTpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "attnOprojTpSize"));
    }
    if (paramJson.contains("attnOprojTpDomain")) {
        attnOprojTpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "attnOprojTpDomain");
    }
    if (paramJson.contains("attnOprojPrefetch")) {
        attnOprojPrefetch = atb_speed::base::FetchJsonParam<bool>(paramJson, "attnOprojPrefetch");
    }
    if (paramJson.contains("hasAttnDp")) {
        hasAttnDp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasAttnDp");
    }
    if (paramJson.contains("attnDpRank")) {
        attnDpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "attnDpRank");
    }
    if (paramJson.contains("attnDpSize")) {
        attnDpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "attnDpSize"));
    }
    if (paramJson.contains("attnDpDomain")) {
        attnDpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "attnDpDomain");
    }
}

void MoeModelParam::ParseParallelParams(const nlohmann::json &paramJson)
{
    if (paramJson.contains("hasMlpTp")) {
        hasMlpTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasMlpTp");
    }
    if (paramJson.contains("mlpTpRank")) {
        mlpTpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "mlpTpRank");
    }
    if (paramJson.contains("mlpTpSize")) {
        mlpTpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "mlpTpSize"));
    }
    if (paramJson.contains("mlpTpDomain")) {
        mlpTpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "mlpTpDomain");
    }
    if (paramJson.contains("hasMoeEp")) {
        hasMoeEp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasMoeEp");
    }
    if (paramJson.contains("moeEpRank")) {
        moeEpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "moeEpRank");
    }
    if (paramJson.contains("moeEpSize")) {
        moeEpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "moeEpSize"));
    }
    if (paramJson.contains("moeEpDomain")) {
        moeEpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "moeEpDomain");
    }
    if (paramJson.contains("hasMoeTp")) {
        hasMoeTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasMoeTp");
    }
    if (paramJson.contains("moeTpRank")) {
        moeTpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "moeTpRank");
    }
    if (paramJson.contains("moeTpSize")) {
        moeTpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "moeTpSize"));
    }
    if (paramJson.contains("moeTpDomain")) {
        moeTpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "moeTpDomain");
    }
    if (paramJson.contains("lmHeadTpRank")) {
        lmHeadTpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "lmHeadTpRank");
    }
    if (paramJson.contains("lmHeadTpSize")) {
        lmHeadTpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "lmHeadTpSize"));
    }
    if (paramJson.contains("lmHeadTpDomain")) {
        lmHeadTpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "lmHeadTpDomain");
    }
    if (paramJson.contains("maxDecodeDpTokenSize")) {
        maxDecodeDpTokenSize = atb_speed::base::FetchJsonParam<int>(paramJson, "maxDecodeDpTokenSize");
    }
    
    // Prefill H3P, Hierarchical & Heterogeneous & Hybrid Parallel
    if (paramJson.contains("enableQkvdownDp")) {
        enableQkvdownDp = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableQkvdownDp");
    }
    if (paramJson.contains("enableSharedExpertDp")) {
        enableSharedExpertDp = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableSharedExpertDp");
    }
    if (paramJson.contains("enableGatingDp")) {
        enableGatingDp = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableGatingDp");
    }
    if (paramJson.contains("enableSharedExpertOverlap")) {
        enableSharedExpertOverlap = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableSharedExpertOverlap");
    }
    if (paramJson.contains("enableLcocTp")) {
        enableLcocTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableLcocTp");
    }
    if (paramJson.contains("enableLcocAll2All")) {
        enableLcocAll2All = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableLcocAll2All");
    }
}

void MoeModelParam::ParseMoEParams(const nlohmann::json &paramJson)
{
    if (paramJson.contains("enableSwiGLUQuantForSharedExperts")) {
        enableSwiGLUQuantForSharedExperts = \
            atb_speed::base::FetchJsonParam<bool>(paramJson, "enableSwiGLUQuantForSharedExperts");
    }
    if (paramJson.contains("enableAllToAllMC2")) {
        enableAllToAllMC2 = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableAllToAllMC2");
    }
    if (paramJson.contains("enableLoadBalance")) {
        enableLoadBalance = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableLoadBalance");
    }
    if (paramJson.contains("enableExtraOprojTp")) {
        enableExtraOprojTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableExtraOprojTp");
    }
    if (paramJson.contains("enableEPWB")) {
        enableEPWB = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableEPWB");
    }
    if (paramJson.contains("numOfRedundantExpert")) {
        numOfRedundantExpert = atb_speed::base::FetchJsonParam<int>(paramJson, "numOfRedundantExpert");
    }
    ParseMoEGateParams(paramJson);
}

void MoeModelParam::ParseMoEGateParams(const nlohmann::json &paramJson)
{
    if (paramJson.contains("moePackQuantType")) {
        this->moePackQuantType = atb_speed::base::FetchJsonParam<int>(paramJson, "moePackQuantType");
    }
    if (paramJson.contains("enableATBGateMatmul")) {
        enableATBGateMatmul = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableATBGateMatmul");
    }
    if (paramJson.contains("routedScalingFactor")) {
        routedScalingFactor = atb_speed::base::FetchJsonParam<float>(paramJson, "routedScalingFactor");
    }
    if (paramJson.contains("scaledTopk")) {
        scaledTopk = atb_speed::base::FetchJsonParam<int>(paramJson, "scaledTopk");
    }
    if (paramJson.contains("enableInitRoutingCutoff")) {
        enableInitRoutingCutoff = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableInitRoutingCutoff");
    }
    if (paramJson.contains("topkGroups")) {
        for (auto item : paramJson["topkGroups"]) {
            topkGroups.push_back(atb_speed::base::FetchJsonParam<int>(item, "topkGroups", true));
        }
    }
    if (paramJson.contains("enableFusedTopk")) {
        enableFusedTopk = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableFusedTopk");
    }
    if (paramJson.contains("enableExpertCumSumOutput")) {
        enableExpertCumSumOutput = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableExpertCumSumOutput");
    }
    if (paramJson.contains("numDanglingSharedExperts")) {
        numDanglingSharedExperts = paramJson.at("numDanglingSharedExperts").get<int64_t>();
    }
}

void MoeModelParam::CheckParallelParamValid()
{
    if (attnTpRank >= attnTpSize) {
        throw std::runtime_error("attnTpSize must be greater than attnTpRank, please check.");
    }
    if (attnDpRank >= attnDpSize) {
        throw std::runtime_error("attnDpSize must be greater than attnDpRank, please check.");
    }
    if (mlpTpRank >= mlpTpSize) {
        throw std::runtime_error("mlpTpSize must be greater than mlpTpRank, please check.");
    }
    if (moeEpRank >= moeEpSize) {
        throw std::runtime_error("moeEpSize must be greater than moeEpRank, please check.");
    }
}

void MoeModelParam::PrintParam()
{
    atb_speed::base::ModelParam::PrintParam();
    ATB_SPEED_LOG_DEBUG(", numOfExperts: " << this->numOfExperts
                  << ", expertParallelDegree: " << this->expertParallelDegree
                  << ", numOfSelectedExperts:" << this->numOfSelectedExperts
                  << ", routingMethod: " << this->routingMethod
                  << ", processLogits" << this->processLogits
                  << ", normHasBias: " << this->normHasBias
                  << ", enableFusedRouting: " << this->enableFusedRouting
                  << ", moeLinearQuantType: " << this->moeLinearQuantType
                  << ", mlpLinearQuantType: " << this->mlpLinearQuantType
                  << ", moeLinearTransposeType: " << this->moeLinearTransposeType
                  << ", mlpLinearTransposeType: " << this->mlpLinearTransposeType
                  << ", hasSharedExpert: " << this->hasSharedExpert
                  << ", hasSharedExpertGate: " << this->hasSharedExpertGate
                  << ", numOfSharedExperts:" << this->numOfSharedExperts
                  << ", firstKDenseReplace: " << this->firstKDenseReplace
                  << ", numOfGroups: " << this->numOfGroups
                  << ", numOfSelectedGroups: " << this->topkGroups);
}

void MoeModelParam::CheckRoutingMethodValid()
{
    std::set<std::string> supportRoutingMethods = {
        "softMaxTopK", "integratedSoftmaxTopK", "deviceLimited", "noAuxTc", "topkFused"};
    if (supportRoutingMethods.find(this->routingMethod) == supportRoutingMethods.end()) {
        std::stringstream ss;
        ss << "The routing method " << this->routingMethod << " is not valid." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
}

void MoeModelParam::CheckProcessLogitsValid()
{
    std::set<std::string> supportProcessLogits = {"none", "normalization", "scaling", "normScaling"};
    if (supportProcessLogits.find(this->processLogits) == supportProcessLogits.end()) {
        std::stringstream ss;
        ss << "The process logits method" << this->processLogits << " is not valid." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
}

void MoeModelParam::CheckParam()
{
    CheckLinearParamsSufficient(this->moeLinearQuantType, this->numHiddenLayers, MOE_LINEAR_TYPE_LENGTH);
    CheckLinearParamsSufficient(this->mlpLinearQuantType, this->numHiddenLayers, MOE_LINEAR_TYPE_LENGTH);
    CheckLinearParamsSufficient(this->moeLinearTransposeType, this->numHiddenLayers, MOE_LINEAR_TYPE_LENGTH);
    CheckLinearParamsSufficient(this->mlpLinearTransposeType, this->numHiddenLayers, MOE_LINEAR_TYPE_LENGTH);
    CheckRoutingMethodValid();
}

MoeDecoderModel::MoeDecoderModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
    this->param.FromString(param);
    this->inTensorCandidates["default_moe"] = {
        "expert_array_model", "expert_group_model", "one_hot_model", "zero_hot_model"};
    this->inTensorCandidates["fused_routing"] = {
        "in_final_hidden_state", "in_final_hidden_state_two", "in_final_bias"};
    this->inTensorCandidates["parallel"] = {
        "in_attn_padding_idx_model", "in_attn_unpadding_idx_model",
        "in_ffn_padding_idx_model", "in_ffn_unpadding_idx_model",
        "in_lm_head_skip_padding_token_indices_model",
        "in_attention_padding_idx_slice", "in_start_expert_idx_model",
        "in_device_expert_count_model",
        "in_lty_idx_model", "in_moe_idx_model", "in_post_lmhead_unpadding_indices"};
}

void MoeDecoderModel::ConstructInTensorMap()
{
    DecoderModel::ConstructInTensorMap();
    atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "default_moe", this->inTensorMap);
    if (this->param.enableSpeculate || this->param.enableSplitFuse || this->param.enablePrefixCache) {
        atb_speed::common::AssignTensorIdx(
            this->inTensorCandidates, "q_len", this->inTensorMap);
    }
    atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "parallel", this->inTensorMap);
}

atb::Status MoeDecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    MoeLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    if (this->param.normType == atb_speed::base::RMS_NORM) {
        MoeDecoderLayer<atb::infer::RmsNormParam> decoderLayer(layerParam);
        CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    } else {
        MoeDecoderLayer<atb::infer::LayerNormParam> decoderLayer(layerParam);
        CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    }
    return atb::NO_ERROR;
}

void MoeDecoderModel::SetLmHeadParam(atb_speed::common::LmHeadParam &lmHeadParam)
{
    lmHeadParam.unpadInputs = this->param.isUnpadInputs;
    lmHeadParam.gatherAhead = this->param.isPrefill || this->param.enablePrefixCache || this->param.hasAttnDp;
    lmHeadParam.hiddenSizePerAttentionHead = this->param.hiddenSizePerAttentionHead;
    lmHeadParam.linearParallelParam.fusionLinearParam.isBF16 = this->param.isBF16;
    lmHeadParam.linearParallelParam.fusionLinearParam.transposeType = this->param.lmHeadTransposeType;
    lmHeadParam.linearParallelParam.fusionLinearParam.matmulBackend = param.matmulBackend;
    lmHeadParam.linearParallelParam.unpadInputs = !this->param.isFA;
    lmHeadParam.linearParallelParam.enableMC2 = this->param.enableMC2;
    lmHeadParam.linearParallelParam.isArgmaxlogits = this->param.enableGreedyPostProcessing;
    lmHeadParam.linearParallelParam.worldSize = this->param.worldSize;
    
    if (this->param.isLmHeadParallel) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
        atb_speed::common::ParallelInfo parallelInfo = this->param.mapping.Get(base::LM_HEAD_TP);
        lmHeadParam.linearParallelParam.tensorParallelInfo.rank = parallelInfo.rank;
        lmHeadParam.linearParallelParam.tensorParallelInfo.worldSize = parallelInfo.rankIds.size();
        if (atb_speed::common::IsA2()) {
            lmHeadParam.linearParallelParam.tensorParallelInfo.backend = "hccl";
            parallelInfo.InitCommDomain(
                lmHeadParam.linearParallelParam.tensorParallelInfo.hcommInfo,
                lmHeadParam.linearParallelParam.tensorParallelInfo.commDomain, "hccl");
        } else {
            lmHeadParam.linearParallelParam.tensorParallelInfo.backend = parallelInfo.defaultBackend;
            parallelInfo.InitCommDomain(
                lmHeadParam.linearParallelParam.tensorParallelInfo.hcommInfo,
                lmHeadParam.linearParallelParam.tensorParallelInfo.commDomain);
        }
    }
}

atb::Status MoeDecoderModel::AddSingleLayer(uint32_t layerId)
{
    atb::Operation *op = nullptr;
    auto attnIt = std::find(this->param.attnSkipLayerSet.begin(), this->param.attnSkipLayerSet.end(), layerId);
    auto mlpIt = std::find(this->param.mlpSkipLayerSet.begin(), this->param.mlpSkipLayerSet.end(), layerId);
    if (attnIt != this->param.attnSkipLayerSet.end() && mlpIt != this->param.mlpSkipLayerSet.end()) {
        return atb::NO_ERROR;
    }
    atb_speed::Model::Node layerNode;
    CHECK_OPERATION_STATUS_RETURN(this->CreateLayerOperation(&op, layerId));

    layerNode.operation.reset(op);
    layerNode.inTensors.resize(layerNode.operation->GetInputNum());
    layerNode.inTensorReshapeFuncs.resize(layerNode.operation->GetInputNum());
    ATB_SPEED_LOG_DEBUG("Layer inputs num is " << layerNode.operation->GetInputNum());
    SetLayerNodeInput(layerNode, layerId);
    ATB_SPEED_LOG_DEBUG("Set layer" << layerId << "'s inputs success.");

    if ((this->param.mapping.Get(base::ATTN_DP).IsEnabled() || this->param.mapping.Get(base::ATTN_CP).IsEnabled())
            && layerId == this->param.numHiddenLayers - 1) {
        layerNode.outTensors = {
            &graph_.internalTensors.at(
                atb_speed::common::GetTensorIdx(this->internalTensorMap, "attn_dp_last_layer"))
        };
    } else {
        layerNode.outTensors = {layerNode.inTensors.at(weightCountPerLayer)};
    }
    if (this->param.enableInterLayerAddNorm && (layerId != (this->param.numHiddenLayers - 1))) {
        layerNode.outTensors.push_back(
            &graph_.internalTensors.at(
                atb_speed::common::GetTensorIdx(this->internalTensorMap, "last_layer_mlp_out")
            )
        );
    }
    graph_.nodes.push_back(layerNode);
    ATB_SPEED_LOG_DEBUG("[+] add base layerNode num" << layerId);
    return atb::NO_ERROR;
}

void MoeDecoderModel::SetLayerParam(MoeLayerParam &layerParam, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    layerParam.numOfExperts = this->param.numOfExperts;
    layerParam.numOfDeviceExperts = this->param.numOfDeviceExperts;
    layerParam.routedScalingFactor = this->param.routedScalingFactor;
    layerParam.expertParallelDegree = this->param.expertParallelDegree;
    layerParam.deviceExpert = this->param.deviceExpert;
    layerParam.routingMethod = this->param.routingMethod;
    layerParam.scaledTopk = this->param.scaledTopk;
    layerParam.enableInitRoutingCutoff = this->param.enableInitRoutingCutoff;
    layerParam.quantGroupSize = this->param.quantGroupSize;
    layerParam.numOfSelectedExperts = this->param.numOfSelectedExperts;
    layerParam.normHasBias = this->param.normHasBias;
    layerParam.enableFusedRouting = this->param.enableFusedRouting;
    layerParam.enableGMMSwigluQuant = this->param.enableGMMSwigluQuant;
    layerParam.enableInitQuant = this->param.enableInitQuant;
    layerParam.enableSwigluQuant = this->param.enableSwigluQuant;
    layerParam.enableFusedTopk = this->param.enableFusedTopk;
    layerParam.enableCVOverlap = this->param.enableCVOverlap;
    layerParam.enableExpertCumSumOutput = param.enableExpertCumSumOutput;
    layerParam.enableAtlasGMMFused = this->param.enableAtlasGMMFused;
    layerParam.processLogits = this->param.processLogits;
    layerParam.hasMoeEp = this->param.hasMoeEp;
    layerParam.isDynamicEp = this->param.expertParallelDegree == 2;
    layerParam.moeLinearQuantType = this->param.moeLinearQuantType[layerId];
    layerParam.mlpLinearQuantType = this->param.mlpLinearQuantType[layerId];
    layerParam.moeLinearTransposeType = this->param.moeLinearTransposeType[layerId];
    layerParam.mlpLinearTransposeType = this->param.mlpLinearTransposeType[layerId];
    layerParam.hasSharedExpert = this->param.hasSharedExpert;
    layerParam.hasSharedExpertGate = this->param.hasSharedExpertGate;
    layerParam.numOfSharedExperts = this->param.numOfSharedExperts;
    layerParam.firstKDenseReplace = this->param.firstKDenseReplace;
    layerParam.numOfGroups = this->param.numOfGroups;
    layerParam.topkGroups = this->param.topkGroups;
    layerParam.moePackQuantType = this->param.moePackQuantType;
    layerParam.enableATBGateMatmul = param.enableATBGateMatmul;
    layerParam.layerId = layerId;
    layerParam.isDenseLayer = layerParam.layerId < layerParam.firstKDenseReplace;
    layerParam.isLastLayer = (layerId == this->param.numHiddenLayers - 1);
    layerParam.enableLoadBalance = this->param.enableLoadBalance;
    layerParam.enableEPWB = this->param.enableEPWB;
    layerParam.numOfRedundantExpert = this->param.numOfRedundantExpert;
    layerParam.numDanglingSharedExperts = this->param.numDanglingSharedExperts;
    layerParam.enableGatingDp = this->param.enableGatingDp && layerParam.layerId >= layerParam.firstKDenseReplace;
    layerParam.enableSharedExpertOverlap = this->param.enableSharedExpertOverlap && this->param.enableSharedExpertDp;
    layerParam.enableLcocAll2All = this->param.enableLcocAll2All;
    layerParam.dispatchAndCombineHcclComm = this->param.dispatchAndCombineHcclComm;
    layerParam.dispatchAndCombinecommDomain = this->param.dispatchAndCombinecommDomain;
    layerParam.enableDispatchCombineV2 = this->param.enableDispatchCombineV2;
    layerParam.enableAllToAllMC2 = this->param.enableAllToAllMC2;
    layerParam.maxDecodeDpTokenSize = this->param.maxDecodeDpTokenSize;
}

atb::Status MoeDecoderModel::AddParallelHostWeight(atb_speed::Model::Node &layerNode, uint32_t &inTensorId)
{
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_attn_padding_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_attn_unpadding_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_ffn_padding_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_ffn_unpadding_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_lm_head_skip_padding_token_indices_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_attention_padding_idx_slice"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_start_expert_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_device_expert_count_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_lty_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_moe_idx_model"));
    return atb::NO_ERROR;
}

void MoeDecoderModel::SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId)
{
    uint32_t inTensorId = 0;
    this->SetLayerNodeDefaultInput(layerNode, layerId, inTensorId);
    this->SetLayerNodeOptionalInput(layerNode, layerId, inTensorId);
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "expert_array_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "expert_group_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "one_hot_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "zero_hot_model"));
    if (this->param.enableSpeculate || this->param.enableSplitFuse || this->param.enablePrefixCache) {
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "q_len"));
    }
    AddParallelHostWeight(layerNode, inTensorId);
}

} // namespace moe
} // namespace atb_speed