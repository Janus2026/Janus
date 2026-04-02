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
#include "atb_speed/utils/singleton.h"
#include "atb_speed/base/external_comm_manager.h"
#include "models/base/param/mapping.h"

namespace atb_speed {
namespace base {

void Mapping::ParseParam(const nlohmann::json &paramJson)
{
    this->worldSize_ = FetchJsonParam<uint32_t>(paramJson, "worldSize");
    this->rank_ = FetchJsonParam<uint32_t>(paramJson, "rank");
    this->rankTableFile_ = FetchJsonParam<std::string>(paramJson, "rankTableFile");
    this->localWorldSize_ = FetchJsonParam<uint32_t>(paramJson, "localWorldSize");
    GetSingleton<ExternalCommManager>().SetLcclCommDomainRange(
        FetchJsonParam<uint32_t>(paramJson, "lcclCommDomainLowerBound"),
        FetchJsonParam<uint32_t>(paramJson, "lcclCommDomainUpperBound")
    );
    std::map<ParallelType, std::string> strategyKeyMap = {
        {WORD_EMBED_TP, "wordEmbedTp"},
        {WORD_EMBED_DP, "wordEmbedDp"},
        {ATTN_TP, "attnTp"},
        {ATTN_DP, "attnDp"},
        {ATTN_CP, "attnCp"},
        {ATTN_INNER_SP, "attnInnerSp"},
        {ATTN_O_PROJ_TP, "attnOProjTp"},
        {ATTN_O_PROJ_DP, "attnOProjDp"},
        {MLP_TP, "mlpTp"},
        {MLP_DP, "mlpDp"},
        {MOE_TP, "moeTp"},
        {MOE_EP, "moeEp"},
        {LM_HEAD_TP, "lmHeadTp"},
        {LM_HEAD_DP, "lmHeadDp"},
        {LCOC_ATTN_TP, "lcocAttnTp"},
    };
    for (auto it = strategyKeyMap.begin(); it != strategyKeyMap.end(); it++) {
        atb_speed::common::ParallelInfo parallelInfo = atb_speed::common::ParallelInfo();
        const nlohmann::json &curParamJson = paramJson[it->second];
        parallelInfo.rank = FetchJsonParam<uint32_t>(curParamJson, "rank");
        parallelInfo.rankIds = FetchJsonParam<std::vector<uint32_t>>(curParamJson["rankIds"], "rankIds", true);
        parallelInfo.bufferSize = FetchJsonParam<uint32_t>(curParamJson, "bufferSize");
        parallelInfo.groupId = FetchJsonParam<uint32_t>(curParamJson, "groupId");
        this->Register(it->first, parallelInfo);
    }
}

void Mapping::InitGlobalCommDomain(std::string defaultBackend)
{
    this->defaultBackend_ = defaultBackend;
    uint32_t streamId = GetSingleton<atb_speed::common::DapManager>().GetStreamId();
    std::vector<uint32_t> rankIds = {};
    for (uint32_t id = 0; id < this->worldSize_; id++) {
        rankIds.push_back(id);
    }
    std::vector<uint32_t> fixedRankIds = rankIds;
    std::string backend = atb_speed::common::InitCommBackend(
        this->localWorldSize_, fixedRankIds, this->defaultBackend_);
    // Create global comm
    ATB_SPEED_LOG_DEBUG("External Comm Manager: InitCommDomain: init");
    GetSingleton<ExternalCommManager>().Init(this->worldSize_, this->rank_,
        backend, this->rankTableFile_, streamId);
    this->isInitialized_ = true;
}

uint32_t Mapping::localWorldSize() const { return localWorldSize_; }

void Mapping::Register(ParallelType parallelType, atb_speed::common::ParallelInfo parallelInfo)
{
    this->parallelStrategies_[parallelType] = parallelInfo;
}

const atb_speed::common::ParallelInfo Mapping::Get(ParallelType parallelType) const
{
    std::stringstream ss;
    auto it = this->parallelStrategies_.find(parallelType);
    if (it == this->parallelStrategies_.end()) {
        ss << "Mapping: Parallel type [" << parallelType << "] is not found. "
           << "Existing strategies are ";
        for (auto item = this->parallelStrategies_.begin(); item != this->parallelStrategies_.end(); item++) {
            ss << item->first << " ";
        }
        throw std::out_of_range(ss.str());
    }

    atb_speed::common::ParallelInfo parallelInfo = it->second;
    std::string backend = atb_speed::common::InitCommBackend(
        this->localWorldSize_, parallelInfo.rankIds, this->defaultBackend_);
    parallelInfo.defaultBackend = backend;

    return parallelInfo;
}

} // namespace base
} // namesapce atb_speed