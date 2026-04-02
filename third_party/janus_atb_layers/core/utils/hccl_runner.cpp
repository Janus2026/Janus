/**
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
#include "atb_speed/utils/hccl_runner.h"
#include <hccl/hccl.h>
#include "atb_speed/log.h"

namespace atb_speed {
HcclRunner::HcclRunner(int rank, int rankSize, int rankRoot)
    : rank_(rank),
      rankSize_(rankSize),
      rankRoot_(rankRoot) {}

HcclRunner::~HcclRunner()
{
    ATB_SPEED_LOG_DEBUG("HcclRunner deconstruct");
}


HcclComm HcclRunner::CreateHcclCommInMulitProcessByRootInfo()
{
    ATB_SPEED_LOG_DEBUG("HCCL Runner single server init ");
    if (!CreateHcclRootInfo()) {
        return nullptr;
    }

    HcclComm newHcclComm = nullptr;
    auto ret = HcclCommInitRootInfo(rankSize_, &hcclRootInfo_, rank_, &newHcclComm);
    if (ret != HCCL_SUCCESS) {
        ATB_SPEED_LOG_ERROR("HcclCommInitRootInfo fail, error:" << ret << ", rank:" << rank_
                       << ", rankSize:" << rankSize_);
    }
    return newHcclComm;
}

bool HcclRunner::CreateHcclRootInfo()
{
    std::string shmName = "hcclShareMem";
    ShareMemory shm(shmName, sizeof(atb_speed::CommInitInfo) + rankSize_ * sizeof(bool));
    auto *shmInfo = static_cast<atb_speed::CommInitInfo *>(shm.GetShm());
    if (!shmInfo) {
        ATB_SPEED_LOG_ERROR("create share memory fail, rank:" << rank_);
        return false;
    }

    // translatedHcclGetRootInfotranslatedhcclRootInfo_(translatedHostIPtranslated), translated,translatedRoortInfo
    // translated,translatedCreateHcclComm
    ATB_SPEED_LOG_DEBUG("create share memory success, rank:" << rank_);
    if (rank_ == rankRoot_) {
        auto ret = HcclGetRootInfo(&hcclRootInfo_);
        if (ret != HCCL_SUCCESS) {
            ATB_SPEED_LOG_ERROR("HcclGetRootInfo fail, error:" << ret << ", rank:" << rank_);
            return false;
        }
        ATB_SPEED_LOG_DEBUG("HcclGetRootInfo success, write to share memory");
        ShmSetHcclRootInfo(shm, *shmInfo);
    } else {
        ATB_SPEED_LOG_DEBUG("get root info from share memory");
        ShmGetHcclRootInfo(shm, *shmInfo);
    }

    return ShmBarrier(shm, *shmInfo);
}

void HcclRunner::ShmGetHcclRootInfo(ShareMemory &shm, const CommInitInfo &shmInfo)
{
    bool commIdReady = false;
    while (!commIdReady) {
        shm.SemLock();
        if (shmInfo.signal != 0) {
            hcclRootInfo_ = shmInfo.hcclRootInfo;
            commIdReady = true;
        }
        shm.SemUnLock();
        if (commIdReady) {
            break;
        }
    }
}

void HcclRunner::ShmSetHcclRootInfo(ShareMemory &shm, CommInitInfo &shmInfo)
{
    shm.SemLock();
    shmInfo.hcclRootInfo = hcclRootInfo_;
    shmInfo.signal = 1;
    shm.SemUnLock();
}

void HcclRunner::ShmSetReady(ShareMemory &shm, CommInitInfo &shmInfo) const
{
    shm.SemLock();
    shmInfo.barrier[rank_] = true;
    shm.SemUnLock();
}

bool HcclRunner::ShmBarrier(ShareMemory &shm, CommInitInfo &shmInfo)
{
    ATB_SPEED_LOG_DEBUG("barrier start, rank:" << rank_);
    ShmSetReady(shm, shmInfo);

    ATB_SPEED_LOG_DEBUG("check all ready start");
    const double timeout = 600;  // 600: 10 minutes timeout
    time_t startTime = time(nullptr);
    bool endSignal = false;
    while (!endSignal) {
        time_t currentTime = time(nullptr);
        if (difftime(currentTime, startTime) > timeout) {
            ATB_SPEED_LOG_ERROR("barrier fail, check all ready timeout");
            endSignal = true;
            return false;
        }

        bool allReady = true;
        shm.SemLock();
        for (int i = 0; i < rankSize_; i++) {
            if (!shmInfo.barrier[i]) {
                allReady = false;
                break;
            }
        }
        shm.SemUnLock();
        if (allReady) {
            ATB_SPEED_LOG_DEBUG("check all ready success");
            break;
        }
    }

    ATB_SPEED_LOG_DEBUG("barrier success, rank:" << rank_);
    return true;
}

}  // namespace atb_speed
