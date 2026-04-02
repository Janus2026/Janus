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
#ifndef ATB_SPEED_PARALLEL_INFO_H
#define ATB_SPEED_PARALLEL_INFO_H
#include <hccl/hccl.h>
#include "atb_speed/log.h"
#include "operations/fusion/utils.h"

namespace atb_speed {
namespace common {
/// Parameters related to parallelism
struct ParallelInfo {
    /// Rank of the device within the communication group
    uint32_t rank = 0;
    /// Size of the communication group
    std::vector<uint32_t> rankIds = {};
    /// Index of the current communication groups
    uint32_t groupId = 0;
    /// Backend communication method
    std::string defaultBackend = "";
    /// The size of the buffer area for sharing data between devices
    uint32_t bufferSize = 0;

    /// Initialize hccl communication handle on demand and get unique communication domain from rankIds
    void InitCommDomain(HcclComm& hcclComm, std::string& commDomain, std::string backend = "") const;
    /// Check if the parallel strategy is enabled
    bool IsEnabled() const;
    /// A summary of the `ParallelInfo` object
    std::string ToString() const;
};

std::string InitCommBackend(uint32_t localWorldSize, const std::vector<uint32_t> rankIds, std::string commBackend);

/// Parameters related to pipeline parallelism
struct PpParallelInfo : public ParallelInfo {
    /// Micro batch size
    int microBatchSize = 1;
    /// Parameters related to the internal tensor parallelism
    ParallelInfo internalTp;
};

} // namespace common
} // namespace atb_speed

#endif