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
#include "atb_speed/utils/share_memory.h"
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <securec.h>
#include <functional>
#include "atb_speed/log.h"
#include "atb_speed/log.h"

constexpr int SEM_TIMEOUT = 300;

ShareMemory::ShareMemory(const std::string &name, uint32_t size) : memSize_(size)
{
    sem_ = sem_open(name.c_str(), O_CREAT, S_IRUSR | S_IWUSR, 1);
    if (SEM_FAILED == sem_) {
        ATB_SPEED_LOG_ERROR("share memory open fail, name:" << name);
        return;
    }
    ATB_SPEED_LOG_DEBUG("create share memory begin, name:" << name);

    SemLock();
    shareMemory_ = (uint8_t *)CreateShareMemory(name, memSize_);
    ATB_SPEED_LOG_DEBUG("create share memory success");
    SemUnLock();
}

void *ShareMemory::GetShm()
{
    return shareMemory_;
};

void ShareMemory::SemLock() const
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += SEM_TIMEOUT;
    int ret = sem_timedwait(sem_, &ts);
    // translated
    if (ret == -1 && errno == ETIMEDOUT) {
        ATB_SPEED_LOG_ERROR("The semaphore waiting duration exceeds 5 minutes. Run the "
            "rm -rf /dev/shm/sem." <<
            fullName_ << " command to clear the semaphore.");
    }
};

void ShareMemory::SemUnLock() const
{
    sem_post(sem_);
};

void *ShareMemory::CreateShareMemory(const std::string &name, uint32_t size)
{
    void *memory = nullptr;
    struct shmid_ds buf;
    key_t key = static_cast<key_t>(std::hash<std::string>{}(name));
    shmid_ = shmget(key, size, IPC_CREAT | 0600); // 0600translated
    ATB_SPEED_LOG_DEBUG("key: " << key << " shmid :" << shmid_);
    if (shmid_ == -1) {
        ATB_SPEED_LOG_ERROR("shmget err, " << "errno is: " <<errno);
        return nullptr;
    }

    memory = shmat(shmid_, nullptr, 0);
    if (memory == reinterpret_cast<void *>(-1)) {
        ATB_SPEED_LOG_ERROR("shmmat err, " << "errno is: " <<errno);
        return nullptr;
    }

    shmctl(shmid_, IPC_STAT, &buf);

    if (buf.shm_nattch == 1) {
        int ret = memset_s(memory, size, 0, size);
        if (ret != EOK) {
            ATB_SPEED_LOG_ERROR("memset_s Error! Error Code: " << ret);
        }
    }
    return memory;
}

ShareMemory::~ShareMemory()
{
    SemLock();
    CleanUpShm();
    SemUnLock();
    CleanUpSem();
}

void ShareMemory::CleanUpShm()
{
    int ret = shmdt(shareMemory_);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR("shmdt failed, " << "errno is: " <<errno);
    }
    shmid_ds buf{};
    shmctl(shmid_, IPC_STAT, &buf);
    if (shmid_ != -1 && buf.shm_nattch == 0) {
        ret = shmctl(shmid_, IPC_RMID, nullptr);
        if (ret != 0) {
            ATB_SPEED_LOG_ERROR("shmid: " << shmid_ << " delete share memory fail(shmctl IPC_RMID failed.) ret: "
            << ret << " errno is: " <<errno);
        }
    }
}

void ShareMemory::CleanUpSem()
{
    int ret = sem_close(sem_);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR("sem_close failed. ret:" << ret);
    }
    ret = sem_unlink(fullName_.c_str());
    if (ret != 0) {
        ATB_SPEED_LOG_INFO("Already unlink sem");
    }
}
