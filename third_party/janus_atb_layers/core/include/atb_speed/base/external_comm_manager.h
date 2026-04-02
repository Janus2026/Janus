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
#ifndef ATB_SPEED_EXTERNAL_COMM_MANAGER_H
#define ATB_SPEED_EXTERNAL_COMM_MANAGER_H

#include <acl/acl.h>
#include <atb/types.h>
#include "hccl/hccl.h"
#include "atb/comm.h"
#include "atb_speed/log.h"

namespace atb_speed {

const std::string LCCL = "lccl";
const std::string HCCL = "hccl";
const std::string LCOC = "lcoc";

/// A cache object contains information of a communication group
class CommInfo {
public:
    ~CommInfo();

    uint64_t cacheId_ = 0;
    uint32_t subCommRankId_ = 0;
    std::vector<uint32_t> rankIds_ = {};
    std::string backend_ = "";
    HcclComm hcclComm_ = nullptr;
    uint32_t bufferSize_ = 0;
    uint32_t streamId_ = 0;
    bool enableReuse_ = true;

    std::string ToString() const;
};

/// A class manages all the communication group (including commDomain and hcclComm ptr)
class ExternalCommManager {
public:
    void Init(uint32_t worldSize, uint32_t subCommRankId,
        std::string backend, std::string rankTableFile, uint32_t streamId);

    void SetLcclCommDomainRange(int32_t lowerBound, int32_t upperBound);

    std::string GetCommDomain(uint32_t groupId, const std::vector<uint32_t> &rankIds,
        uint32_t subCommRankId, std::string backend, uint32_t bufferSize, uint32_t streamId,
        bool enableReuse = true);

    HcclComm GetCommPtr(std::string commDomain);

    std::string PrintCommInfo();

    uint32_t worldSize_ = 0;
    uint32_t rank_;
    std::string rankTableFile_ = "";

private:
    std::string GetCommDomainFromCache(
        const std::vector<uint32_t> &rankIds, std::string backend, uint32_t bufferSize, uint32_t streamId);
    std::string GetSelfAssignedCommDomain(std::shared_ptr<CommInfo> &commInfo, uint32_t groupId);
    std::string GetHcclSubCommDomain(std::shared_ptr<CommInfo> &commInfo, uint32_t groupId);
    std::string GetHcclGlobalCommDomain(std::shared_ptr<CommInfo> &commInfo);

    std::map<std::string, std::shared_ptr<CommInfo>> commInfoCache_ = {};
    HcclComm globalComm_ = nullptr;
    uint32_t commDomainCounter_ = 0;
    uint32_t lcclCommDomainLowerBound_ = 0;
    uint32_t lcclCommDomainUpperBound_ = 0;
};

}  // namespace atb_speed

#endif  // ATB_SPEED_EXTERNAL_COMM_MANAGER_H