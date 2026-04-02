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
#ifndef ATB_SPEED_EVENT_MANAGER_H
#define ATB_SPEED_EVENT_MANAGER_H

#include <acl/acl.h>
#include <atb/context.h>
#include <atb/operation.h>
#include <atb/atb_infer.h>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <map>
#include <thread>
#include <atomic>
#include <nlohmann/json.hpp>

#include "atb_speed/log.h"

namespace atb_speed {
extern thread_local std::vector<std::pair<atb::Operation*, atb::common::EventParam>> g_eventOperationsOfModel;

// translatedEventManagertranslated:
enum EventManagerStatus {
    EM_SUCCESS = 0, // translated
    EM_CREATE_EVENT_FAILED = 1, // translated ACL translated
    EM_PUSH_EVENT_FAILED = 2, // translated
    EM_POP_EVENT_FAILED = 3, // translated
    EM_POP_EVENT_TIMEOUT = 4, // translated
    EM_INVALID_ACTION = 5, // translated(translated PUSH translated POP)
    EM_INVALID_TYPE = 6, // translated(translated RECORD translated WAIT)
    EM_DESTROY_EVENT_FAILED = 7, // translated ACL translated
    EM_OPERATION_CREATION_FAILED = 8, // translated/translated(translated atb::CreateOperation translated)
    EM_INVALID_KEY = 9, // translated
    EM_UNKNOWN_ERROR = 10, // translated,translated
};

// translated:
enum class EventType {
    UNDEFINED, // translated
    RECORD,    // translated
    WAIT,      // translated
};

// translated:
enum class EventAction {
    PUSH, // translated
    POP,  // translated
};

class EventManager {
public:
    // translated,translated EventManager translated
    static EventManager& GetInstance();

    EventManager(const EventManager&) = delete;
    EventManager& operator=(const EventManager&) = delete;

    /**
     * @brief translated ACL translated
     * @param timeout translated(translated:translated)
     */
    void SetWaitOperationTimeout(uint32_t timeout);

    //======================================================
    // [translated]
    // - RecordEvent: translated,translated PUSH translated,translated POP translated.
    // - WaitEvent: translated,translated PUSH translated,translated POP translated.
    //------------------------------------------------------

    /**
     * @brief translated,translated,translated eventAction(PUSH/POP)translated.
     * @param op translated,translated
     * @param eventAction translated(PUSH translated POP)
     * @param pipeKey translated(translated "default")
     * @return translated(atb::Status)
     */
    atb::Status RecordEvent(atb::Operation*& op, EventAction eventAction, const std::string &pipeKey = "default");

    /**
     * @brief translated,translated,translated eventAction(PUSH/POP)translated.
     * @param op translated,translated
     * @param eventAction translated(PUSH translated POP)
     * @param pipeKey translated(translated "default")
     * @return translated(atb::Status)
     */
    atb::Status WaitEvent(atb::Operation*& op, EventAction eventAction, const std::string &pipeKey = "default");

private:
    // translated private,translated
    // translated ACL translated,translated 180 translated
    EventManager();
    ~EventManager();

    /**
     * @brief translated ACL translated(translated ACL_EVENT_SYNC translated)
     * @param event translated
     * @return EM_SUCCESS translated EM_CREATE_EVENT_FAILED
     */
    EventManagerStatus CreateEvent(aclrtEvent &event) const;

    /**
     * @brief translated
     * @param event translated
     * @return EM_SUCCESS translated EM_DESTROY_EVENT_FAILED
     */
    EventManagerStatus DestroyEvent(aclrtEvent event) const;

    /**
     * @brief translated
     * @param event translated event,translated
     * @param pipeKey translated
     * @return EM_SUCCESS translated
     */
    EventManagerStatus CreateAndPushEvent(aclrtEvent &event, const std::string &pipeKey);

    /**
     * @brief translated
     * @param event translated
     * @param pipeKey translated
     * @return EM_SUCCESS translated EM_POP_EVENT_TIMEOUT
     */
    EventManagerStatus PopEvent(aclrtEvent &event, const std::string &pipeKey);

    /**
     * @brief translated,translated/translated
     * translated eventAction(PUSH/POP)translated eventType(RECORD/WAIT)translated ACL translated,translated(eventParam).
     * @param eventAction translated(PUSH translated POP)
     * @param eventType translated(RECORD translated WAIT)
     * @param op translated,translated
     * @param pipeKey translated
     * @return translated(atb::Status)
     */
    atb::Status EventInternal(EventAction eventAction,
                              EventType eventType,
                              atb::Operation*& op,
                              const std::string &pipeKey);

private:
    // translated,translated,translated
    std::condition_variable eventCond_;
    // translated:translated push/pop translated
    std::map<std::string, std::queue<aclrtEvent>> eventQueues_;
    std::map<std::string, std::queue<std::pair<atb::Operation*, atb::common::EventParam>>> opsWithoutEvent_;
    // translated eventQueue_ translated
    std::mutex queueMutex_;
    // translated event translated
    std::atomic<uint64_t> eventCount_{0};
};
}  // namespace atb_speed

#endif  // ATB_SPEED_EVENT_MANAGER_H