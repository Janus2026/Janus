/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#include "workspace.h"
#include "atb_speed/log.h"
#include "buffer_device.h"

namespace atb_speed {

Workspace::Workspace()
{
    workspaceBuffer_[0].reset(new BufferDevice(629145600)); // 629145600 translated
}

Workspace::~Workspace() {}

void Workspace::ClearCache()
{
    for (auto& buff : std::as_const(workspaceBuffer_)) {
        buff.second->ClearBuffer();
    }
}

int32_t Workspace::GetCachedNum()
{
    int32_t rt = 0;
    for (auto& buff : std::as_const(workspaceBuffer_)) {
        rt = rt + buff.second->GetCachedNum();
    }
    return rt;
}

void *Workspace::GetWorkspaceBuffer(uint64_t bufferSize, uint32_t bufferKey)
{
    if (workspaceBuffer_.count(bufferKey) == 0) {
        workspaceBuffer_[bufferKey].reset(new BufferDevice(bufferSize));
    }
    return workspaceBuffer_[bufferKey]->GetBuffer(bufferSize);
}

} // namespace atb_speed