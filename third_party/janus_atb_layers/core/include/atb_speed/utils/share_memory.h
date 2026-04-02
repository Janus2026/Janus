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
#ifndef SHAREMEMORY_H
#define SHAREMEMORY_H

#include <semaphore.h>
#include <cstdint>
#include <iostream>

class ShareMemory {
public:
    ShareMemory(const std::string &name, uint32_t size);
    ~ShareMemory();
    ShareMemory(const ShareMemory &other) = delete;
    ShareMemory &operator=(const ShareMemory &other) = delete;
    void *GetShm();
    void SemLock() const;
    void SemUnLock() const;

private:
    void *CreateShareMemory(const std::string &name, uint32_t size);
    void CleanUpShm();
    void CleanUpSem();

private:
    std::string fullName_;
    sem_t *sem_ = nullptr;
    uint8_t *shareMemory_ = nullptr;
    uint32_t memSize_ = 0;
    int shmid_ = -1;
};

#endif
