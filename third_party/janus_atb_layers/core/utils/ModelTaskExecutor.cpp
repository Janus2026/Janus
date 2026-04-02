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
#include <acl/acl.h>

#include "atb_speed/utils/ModelTaskExecutor.h"

namespace atb_speed {
ModelTaskExecutor::~ModelTaskExecutor()
{
    for (auto &worker : workers_) {
        auto task = [&worker]() -> int {
            worker.stop = true;
            return 0;
        };
        worker.queue.Enqueue(task);
        worker.thread.join();
    }
}

void ModelTaskExecutor::PushTask(int idx, const Task &task)
{
    auto it = idx2worker_.find(idx);
    if (it == idx2worker_.end()) {
        std::lock_guard<std::mutex> guard(mutex_);
        it = idx2worker_.find(idx);
        if (it == idx2worker_.end()) {
            uint32_t workerId = workers_.size();
            workers_.emplace_back();
            auto &worker = workers_[workerId];
            worker.deviceIdx = idx;
            worker.thread = std::thread(&ModelTaskExecutor::WorkerThread, this, workerId);
            it = idx2worker_.insert({idx, workerId}).first;
        }
    }
    auto &worker = workers_[it->second];
    worker.queue.Enqueue(task);
    return;
}

void ModelTaskExecutor::WorkerThread(int workerId)
{
    ATB_SPEED_LOG_DEBUG("WorkerThread " << workerId << " start.");
    auto &worker = workers_[workerId];
    int ret = aclrtSetDevice(worker.deviceIdx);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR("AsdRtDeviceSetCurrent fail, error:" << ret);
    }
    while (!worker.stop) {
        auto task = worker.queue.Dequeue();
        task();
    }
    ATB_SPEED_LOG_DEBUG("WorkerThread " << workerId << " end.");
    return;
}
} // namespace atb_speed