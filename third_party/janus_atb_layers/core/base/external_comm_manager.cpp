/*
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

#include <cmath>
#include <cerrno>
#include "securec.h"
#include "atb_speed/base/external_comm_manager.h"

namespace atb_speed {

CommInfo::~CommInfo()
{
    ATB_SPEED_LOG_DEBUG("External Comm Manager: CommInfo ["
        << std::hash<const CommInfo*>{}(this) << "] destruction starts.");
    if (this->hcclComm_ != nullptr) {
        auto ret = HcclCommDestroy(this->hcclComm_);
        if (ret != HCCL_SUCCESS) {
            ATB_SPEED_LOG_ERROR("External Comm Manager: Call `HcclCommDestroy` API from CANN "
                << "to destroy hccl communication group failed. "
                << "Error code: " << ret << ". "
                << "Check the default log path at $HOME/ascecnd/log for more details. ");
        }
    }
    this->hcclComm_ = nullptr;
}

std::string CommInfo::ToString() const
{
    std::stringstream ss;
    ss << "Cache Addr[" << this << "] cacheId_: " << cacheId_
    << ", subCommRankId_: " << subCommRankId_
    << ", rankIds_: " << rankIds_
    << ", bufferSize_: " << bufferSize_
    << ", backend_: " << backend_
    << ", hcclComm_: " << hcclComm_
    << ", streamId_: " << streamId_;
    return ss.str();
}

bool AreVectorsEqual(const std::vector<uint32_t> &rankIdsA, const std::vector<uint32_t> &rankIdsB)
{
    if (rankIdsA.size() != rankIdsB.size()) {
        return false;
    }
    for (size_t i = 0; i < rankIdsA.size(); i++) {
        if (rankIdsA.at(i) != rankIdsB.at(i)) {
            return false;
        }
    }
    return true;
}

void ExternalCommManager::Init(uint32_t worldSize, uint32_t subCommRankId,
    std::string backend, std::string rankTableFile, uint32_t streamId)
{
    ATB_SPEED_LOG_DEBUG("External Comm Manager: try to create global comm with "
        << "worldSize " << worldSize << ", subCommRankId " << subCommRankId
        << ", backend " << backend << ", rankTableFile " << rankTableFile
    );

    if (this->globalComm_ != nullptr) {
        ATB_SPEED_LOG_DEBUG("External Comm Manager: A global communication group is already created, "
            << "so the creation process will be skipped.");
        return;
    }

    this->worldSize_ = worldSize;
    this->rank_ = subCommRankId;
    this->rankTableFile_ = rankTableFile;

    std::vector<uint32_t> rankIds = {};
    for (uint32_t id = 0; id < worldSize; id++) {
        rankIds.push_back(id);
    }
    std::shared_ptr<CommInfo> commInfo = std::make_shared<CommInfo>();
    commInfo->cacheId_ = this->commInfoCache_.size();
    commInfo->subCommRankId_ = subCommRankId;
    commInfo->rankIds_ = rankIds;
    commInfo->backend_ = backend;
    commInfo->streamId_ = streamId;

    std::string commDomain = "";
    if ((backend == HCCL || this->rankTableFile_ != "") && rankIds.size() > 1) {
        commDomain = GetHcclGlobalCommDomain(commInfo);
    } else if ((backend == LCCL || backend == LCOC) && rankIds.size() > 1) {
        commDomain = GetCommDomainFromCache(rankIds, backend, 200, streamId);  // 200: buffer size
        if (commDomain == "") { commDomain = GetSelfAssignedCommDomain(commInfo, 0); }
    }
    ATB_SPEED_LOG_DEBUG("External Comm Manager: Add [" << commDomain << "] to cache.");
    this->commInfoCache_[commDomain] = commInfo;
}

void ExternalCommManager::SetLcclCommDomainRange(int32_t lowerBound, int32_t upperBound)
{
    this->lcclCommDomainLowerBound_ = lowerBound;
    this->lcclCommDomainUpperBound_ = upperBound;
}

std::string ExternalCommManager::GetCommDomain(uint32_t groupId, const std::vector<uint32_t> &rankIds,
    uint32_t subCommRankId, std::string backend, uint32_t bufferSize, uint32_t streamId, bool enableReuse)
{
    ATB_SPEED_LOG_DEBUG("External Comm Manager: try to create comm with rankIds " << rankIds
        << ", subCommRankId " << subCommRankId << ", backend: " << backend << ", bufferSize " << bufferSize
        << ", streamId " << streamId);

    std::string commDomain = "";

    if (rankIds.size() <= 1) { return commDomain; }

    if (enableReuse) {
        ATB_SPEED_LOG_DEBUG("External Comm Manager: try to reuse communication group from cache.");
        commDomain = GetCommDomainFromCache(rankIds, backend, bufferSize, streamId);
        if (commDomain != "") {
            return commDomain;
        }
    }

    std::shared_ptr<CommInfo> commInfo = std::make_shared<CommInfo>();
    commInfo->cacheId_ = this->commInfoCache_.size();
    commInfo->subCommRankId_ = subCommRankId;
    commInfo->rankIds_ = rankIds;
    commInfo->backend_ = backend;
    commInfo->bufferSize_ = bufferSize;
    commInfo->streamId_ = streamId;
    commInfo->enableReuse_ = enableReuse;
    if ((backend == LCCL || backend == LCOC) && rankIds.size() > 1) {
        commDomain = GetSelfAssignedCommDomain(commInfo, groupId);
    } else if (backend == HCCL && rankIds.size() > 1) {
        commDomain = GetHcclSubCommDomain(commInfo, groupId);
    }
    this->commInfoCache_[commDomain] = commInfo;
    ATB_SPEED_LOG_DEBUG("External Comm Manager: Add [" << commDomain << "] to cache");
    return commDomain;
}

std::string ExternalCommManager::GetCommDomainFromCache(
    const std::vector<uint32_t> &rankIds, std::string backend, uint32_t bufferSize, uint32_t streamId)
{
    std::map<std::string, std::shared_ptr<CommInfo>>::iterator it;
    for (it = this->commInfoCache_.begin(); it != this->commInfoCache_.end(); it++) {
        if (AreVectorsEqual(it->second->rankIds_, rankIds) && \
            it->second->backend_ == backend && it->second->bufferSize_ == bufferSize && \
            it->second->streamId_ == streamId && it->second->enableReuse_
        ) {
            ATB_SPEED_LOG_DEBUG("External Comm Manager: Comm with rankIds " << rankIds
                << ", bufferSize " << bufferSize << ", backend: " << backend
                << ", streamId" << streamId << " hit. CommDomain [" << it->first << "] is reused.");
            return it->first;
        }
    }
    return "";
}

std::string ExternalCommManager::GetSelfAssignedCommDomain(std::shared_ptr<CommInfo> &commInfo, uint32_t groupId)
{
    uint32_t commDomainInt = this->lcclCommDomainLowerBound_ + this->commDomainCounter_ + groupId;
    if (commDomainInt >= this->lcclCommDomainUpperBound_) {
        std::stringstream ss;
        ss << "External Comm Manager: Lccl commDomain exceeds the upper bound. "
            << "Available commDomain range is [" << this->lcclCommDomainLowerBound_
            << ", " << this->lcclCommDomainUpperBound_ << "]. "
            << "The range of the communication domain is determinded by `num_lccl_comm_shards` "
            << "and `lccl_comm_shard_id`. Please review initializaion parameters "
            << "of the `GeneratorTorch` object.";
        throw std::runtime_error(ss.str());
    }
    std::string commDomain = std::to_string(commDomainInt);
    this->commDomainCounter_ = this->commDomainCounter_ + ceil(this->worldSize_ / commInfo->rankIds_.size());
    ATB_SPEED_LOG_DEBUG("External Comm Manager: commDomainCounter_ update to " << this->commDomainCounter_);
    return commDomain;
}

std::string ExternalCommManager::GetHcclGlobalCommDomain(std::shared_ptr<CommInfo> &commInfo)
{
    ATB_SPEED_LOG_DEBUG("GetHcclGlobalCommDomain start.");
    std::string commDomain = "";
    if (this->rankTableFile_ != "") {
        char commName[128] = {};  // 128: max commName length
        this->globalComm_ = atb::Comm::CreateHcclCommByRankTableFile(commInfo->subCommRankId_, this->worldSize_,
            this->rankTableFile_.data(), commName);
        if (this->globalComm_ == nullptr) {
            throw std::runtime_error("External Comm Manager: Create the hccl communication group failed. " \
                "export ASDOPS_LOG_LEVEL=ERROR, export ASDOPS_LOG_TO_STDOUT=1 to see more details. " \
                "Default log path is $HOME/atb/log. ");
        }
        commInfo->hcclComm_ = this->globalComm_;
        char hcclCommName[128] = {};
        HcclGetCommName(this->globalComm_, hcclCommName);
        commDomain = std::string(hcclCommName);
    } else {
        // There is only one global commonDomain. Thus, group Id is 0.
        commDomain = GetSelfAssignedCommDomain(commInfo, 0);
    }
    ATB_SPEED_LOG_DEBUG("GetHcclGlobalCommDomain end.");

    return commDomain;
}

std::string ExternalCommManager::GetHcclSubCommDomain(std::shared_ptr<CommInfo> &commInfo, uint32_t groupId)
{
    ATB_SPEED_LOG_DEBUG("GetHcclSubCommDomain start.");
    std::string commDomain = "";
    if (this->globalComm_ != nullptr) {
        HcclComm hcclComm;
        HcclCommConfig config;
        HcclCommConfigInit(&config);
        config.hcclBufferSize = commInfo->bufferSize_;
        std::vector<uint32_t> tempRankIds = {};
        for (auto item : commInfo->rankIds_) { tempRankIds.push_back(item); }
        auto ret = HcclCreateSubCommConfig(&this->globalComm_, tempRankIds.size(), tempRankIds.data(),
            commInfo->cacheId_, commInfo->subCommRankId_, &config, &hcclComm);
        if (hcclComm == nullptr) {
            ATB_SPEED_LOG_ERROR("External Comm Manager: Call `HcclCreateSubCommConfig` API from CANN "
                << "to create the hccl communication group failed. "
                << "Error code: " << ret << ". "
                << "Check the default log path at $HOME/ascecnd/log for more details. ");
        }
        commInfo->hcclComm_ = hcclComm;
        char hcclCommName[128] = {};
        HcclGetCommName(hcclComm, hcclCommName);
        commDomain = std::string(hcclCommName);
    } else {
        commDomain = GetSelfAssignedCommDomain(commInfo, groupId);
    }
    ATB_SPEED_LOG_DEBUG("GetHcclSubCommDomain end.");
    return commDomain;
}

HcclComm ExternalCommManager::GetCommPtr(std::string commDomain)
{
    if (commDomain == "") { return nullptr; }
    auto it = this->commInfoCache_.find(commDomain);
    if (it == this->commInfoCache_.end()) {
        std::stringstream ss;
        ss << "External Comm Manager: Comm domain[" << commDomain << "] not found in cache.";
        throw std::out_of_range(ss.str());
    }
    return it->second->hcclComm_;
}

std::string ExternalCommManager::PrintCommInfo()
{
    std::stringstream ss;
    ss << "External Comm Manager: Comm Info Cache Summary: Count " << this->commInfoCache_.size();
    std::map<std::string, std::shared_ptr<CommInfo>>::const_iterator it;
    for (it = this->commInfoCache_.begin(); it != this->commInfoCache_.end(); it++) {
        ss << " Comm domain[" << it->first << "] " << it->second->ToString();
    }
    return ss.str();
}

}  // namespace atb_speed