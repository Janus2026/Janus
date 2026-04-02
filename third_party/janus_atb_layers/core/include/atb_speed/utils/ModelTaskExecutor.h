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

#ifndef ATB_SPEED_MODEL_TASK_EXECUTOR_H
#define ATB_SPEED_MODEL_TASK_EXECUTOR_H

#include <acl/acl.h>
#include <thread>
#include <mutex>
#include <deque>
#include <map>
#include <functional>

#include "atb_speed/utils/TaskQueue.h"
#include "atb_speed/log.h"

namespace atb_speed {

class ModelTaskExecutor {
public:
    struct Worker {
        bool stop = false;
        std::thread thread;
        TaskQueue queue;
        int deviceIdx = -1;
    };

public:
    static ModelTaskExecutor& Instance()
    {
        static ModelTaskExecutor instance;
        return instance;
    }

public:
    ~ModelTaskExecutor();

    void PushTask(int idx, const Task &task);

private:
    ModelTaskExecutor() {}

    void WorkerThread(int workerId);

private:
    std::mutex mutex_;
    std::deque<Worker> workers_;
    std::map<int, uint32_t> idx2worker_;
};
}  // namespace atb_speed

#endif  // ATB_SPEED_MODEL_TASK_EXECUTOR_H
