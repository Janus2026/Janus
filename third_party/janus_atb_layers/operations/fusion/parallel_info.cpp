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
#include "parallel_info.h"

namespace atb_speed {
namespace common {

std::string InitCommBackend(uint32_t localWorldSize, const std::vector<uint32_t> rankIds, std::string commBackend)
{
    if (localWorldSize <= 0) {
        throw std::runtime_error("Number of devices in the current node is less than or equal to 0.");
    }
    // Get backend
    std::string backend = commBackend;
    // change to hccl if the communication channel across nodes
    int32_t currentDevice = -1;
    for (uint32_t item : rankIds) {
        if (currentDevice != -1 && static_cast<int32_t>(ceil(item / localWorldSize)) != currentDevice) {
            backend = "hccl";
            break;
        }
        currentDevice = static_cast<int32_t>(ceil(item / localWorldSize));
    }
    // The hccl backend is utilized in the single node scenario
    // when a rankTableFile is supplied and the communication channel spans the entire world size.
    uint32_t worldSize = GetSingleton<ExternalCommManager>().worldSize_;
    if (worldSize <= localWorldSize && GetSingleton<ExternalCommManager>().rankTableFile_ != "" && \
        rankIds.size() == worldSize) {
        backend = "hccl";
    }
    return backend;
}

void ParallelInfo::InitCommDomain(HcclComm& hcclComm, std::string& commDomain, std::string backend) const
{
    if (backend == "") {
        backend = this->defaultBackend;
    }
    // Get current stream id
    uint32_t streamId = GetSingleton<DapManager>().GetStreamId();

    // Assign commDomain by rankIds and rank
    commDomain = GetSingleton<ExternalCommManager>().GetCommDomain(
        this->groupId, this->rankIds, this->rank, backend, this->bufferSize, streamId);
    // Get hcclComm (only created when hccl backend is used and inference across multi nodes)
    hcclComm = GetSingleton<ExternalCommManager>().GetCommPtr(commDomain);

    ATB_SPEED_LOG_DEBUG(this->ToString());
}

bool ParallelInfo::IsEnabled() const
{
    return this->rankIds.size() > 1;
}

std::string ParallelInfo::ToString() const
{
    std::stringstream ss;
    ss << "ParallelInfo: rank: " << this->rank
        << ", rankIds: " << this->rankIds
        << ", groupId: " << this->groupId
        << ", defaultBackend: " << this->defaultBackend
        << ", bufferSize: " << this->bufferSize;
    return ss.str();
}

} // namespace common
} // namesapce atb_speed