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
#ifndef ATB_SPEED_HCCL_RUNNER_H
#define ATB_SPEED_HCCL_RUNNER_H
#include <hccl/hccl_types.h>
#include "share_memory.h"

namespace atb_speed {
struct CommInitInfo {
    int signal = 0;
    HcclRootInfo hcclRootInfo = {};
    bool barrier[1]; // Flexible array member
};

class HcclRunner {
public:
    explicit HcclRunner(int rank = 0, int rankSize = 0, int rankRoot = 0);
    ~HcclRunner();
    HcclComm CreateHcclCommInMulitProcessByRootInfo();

protected:
    int rank_ = 0;
    int rankSize_ = 0;
    int rankRoot_ = 0;
    HcclRootInfo hcclRootInfo_ = {};

private:
    bool CreateHcclRootInfo();
    void ShmGetHcclRootInfo(ShareMemory &shm, const CommInitInfo &shmInfo);
    void ShmSetHcclRootInfo(ShareMemory &shm, CommInitInfo &shmInfo);
    bool ShmBarrier(ShareMemory &shm, CommInitInfo &shmInfo);
    void ShmSetReady(ShareMemory &shm, CommInitInfo &shmInfo) const;
};
}  // namespace atb_speed
#endif