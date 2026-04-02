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

#include "operations/fusion/utils.h"
#include "atb_speed/base/event_manager.h"
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/utils/singleton.h"

#include <gflags/gflags.h>
DEFINE_bool(enable_atb_comm_multiprocess,
            false,
            "whether to use multiprocess mode.");
            
namespace atb_speed {
namespace common {

const std::string CMO_COMPUTE = "cmo_compute";
const std::string CMO_OPROJ = "cmo_oproj";
const std::string CMO_MLAPO = "cmo_mlapo";
const std::string CV_START = "cv_start";
const std::string VECTOR_CONTROL = "vector_control";
const std::string CUBE_CONTROL = "cube_control";
const std::string COMPUTE_EVENT = "compute";
const std::string COMM_EVENT = "comm";
const std::string END_EVENT = "end";
const std::string CC_START = "cc_start";
const std::string COMM_CONTROL = "comm_control";
const std::string COMP_CONTROL = "compute_control";

void DapManager::SetRole(DapRole role) { this->currentRole = role; }

DapRole DapManager::GetRole() const { return this->currentRole; }

std::string DapManager::GetSuccessorSuffix() const
{
    return "_successor";
}

uint32_t DapManager::GetStreamId() const
{
    return this->currentRole == DapRole::SUCCESSOR ? 1 : 0;
}

int32_t CommOpCounter::Increment()
{
    DapRole currentRole = GetSingleton<common::DapManager>().GetRole();
    std::map<DapRole, int32_t>::iterator it = this->count.find(currentRole);
    if (it == this->count.end()) {
        this->count[currentRole] = 1;
        return 1;
    }

    int &currentRoleCount = it->second;
    currentRoleCount += 1;
    return currentRoleCount;
}

int32_t CommOpCounter::GetCount()
{
    DapRole currentRole = GetSingleton<common::DapManager>().GetRole();
    std::map<DapRole, int32_t>::iterator it = this->count.find(currentRole);
    if (it == this->count.end()) {
        return 0;
    }
    int &currentRoleCount = it->second;
    return currentRoleCount;
}

void CommOpCounter::Reset()
{
    std::map<DapRole, int32_t>::iterator it;
    for (it = this->count.begin(); it != this->count.end(); it++) {
        it->second = 0;
    }
}

atb::Status AddDapEventsBeforeComm(atb::GraphParam &opGraph)
{
    DapRole dapRole = GetSingleton<common::DapManager>().GetRole();
    atb_speed::EventAction actionType =
        dapRole == DapRole::PRECEDER ? atb_speed::EventAction::PUSH : atb_speed::EventAction::POP;
    std::stringstream ss;
    std::string role = dapRole == DapRole::PRECEDER ? "PRECEDER" : "SUCCESSOR";
    std::string action = actionType == atb_speed::EventAction::PUSH ? "PUSH" : "POP";
    if (dapRole != DapRole::UNDEFINED_ROLE) {
        atb::Node computeRecordNode;
        computeRecordNode.inTensorIds = {};
        computeRecordNode.outTensorIds = {};
        CHECK_OPERATION_STATUS_RETURN(
            atb_speed::EventManager::GetInstance().RecordEvent(computeRecordNode.operation, actionType, COMPUTE_EVENT));
        opGraph.nodes.push_back(computeRecordNode);
        ss.str("");
        ss << "[Events] [" << role << "] [" << action << "] [RECORD] [COMPUTE]";
        ATB_SPEED_LOG_DEBUG(ss.str());

        if (!(dapRole == DapRole::PRECEDER && GetSingleton<atb_speed::common::CommOpCounter>().GetCount() == 0)) {
            atb::Node commWaitNode;
            commWaitNode.inTensorIds = {};
            commWaitNode.outTensorIds = {};
            CHECK_OPERATION_STATUS_RETURN(
                atb_speed::EventManager::GetInstance().WaitEvent(commWaitNode.operation, actionType, COMM_EVENT));
            opGraph.nodes.push_back(commWaitNode);
            ss.str("");
            ss << "[Events] [" << role << "] [" << action << "] [WAIT] [COMM]";
            ATB_SPEED_LOG_DEBUG(ss.str());
        }
    }
    return atb::NO_ERROR;
};

atb::Status AddDapEventsAfterComm(atb::GraphParam &opGraph)
{
    DapRole dapRole = GetSingleton<common::DapManager>().GetRole();
    atb_speed::EventAction actionType =
        dapRole == DapRole::PRECEDER ? atb_speed::EventAction::PUSH : atb_speed::EventAction::POP;
    std::stringstream ss;
    std::string role = dapRole == DapRole::PRECEDER ? "PRECEDER" : "SUCCESSOR";
    std::string action = actionType == atb_speed::EventAction::PUSH ? "PUSH" : "POP";
    if (dapRole != DapRole::UNDEFINED_ROLE) {
        atb::Node commRecordNode;
        commRecordNode.inTensorIds = {};
        commRecordNode.outTensorIds = {};
        CHECK_OPERATION_STATUS_RETURN(
            atb_speed::EventManager::GetInstance().RecordEvent(commRecordNode.operation, actionType, COMM_EVENT));
        opGraph.nodes.push_back(commRecordNode);
        ss.str("");
        ss << "[Events] [" << role << "] [" << action << "] [RECORD] [COMM]";
        ATB_SPEED_LOG_DEBUG(ss.str());

        atb::Node computeWaitNode;
        computeWaitNode.inTensorIds = {};
        computeWaitNode.outTensorIds = {};
        CHECK_OPERATION_STATUS_RETURN(
            atb_speed::EventManager::GetInstance().WaitEvent(computeWaitNode.operation, actionType, COMPUTE_EVENT));
        opGraph.nodes.push_back(computeWaitNode);
        ss.str("");
        ss << "[Events] [" << role << "] [" << action << "] [WAIT] [COMPUTE]";
        ATB_SPEED_LOG_DEBUG(ss.str());
    }

    GetSingleton<atb_speed::common::CommOpCounter>().Increment();
    return atb::NO_ERROR;
};

void AssignTensorIdx(
    const std::map<std::string, std::vector<std::string>> &tensorCandidates,
    std::string targetKey, uint32_t &tensorIdx, std::map<std::string, uint32_t> &tensorMap)
{
    if (tensorCandidates.find(targetKey) == tensorCandidates.end()) {
        ATB_SPEED_LOG_WARN("targetKey: " << targetKey << " not found in tensorCandidates");
        return;
    }

    for (std::string tensor : tensorCandidates.at(targetKey)) {
        tensorMap[tensor] = tensorIdx;
        tensorIdx++;
    }
}

void AssignTensorIdx(
    const std::map<std::string, std::vector<std::string>> &tensorCandidates,
    std::string targetKey, std::map<std::string, uint32_t> &tensorMap)
{
    if (tensorCandidates.find(targetKey) == tensorCandidates.end()) {
        ATB_SPEED_LOG_WARN("targetKey: " << targetKey << " not found in tensorCandidates");
        return;
    }

    uint32_t startTensorIdx = tensorMap.size();
    for (std::string tensor : tensorCandidates.at(targetKey)) {
        tensorMap[tensor] = startTensorIdx;
        startTensorIdx++;
    }
}

template <typename T>
void AddTensorToList(
    const std::map<std::string, T> &tensorCandidates,
    std::string targetKey, T &tensorList)
{
    if (tensorCandidates.find(targetKey) == tensorCandidates.end()) {
        ATB_SPEED_LOG_WARN("targetKey: " << targetKey << " not found in tensorCandidates");
        return;
    }

    for (const auto& item : tensorCandidates.at(targetKey)) {
        tensorList.push_back(item);
    }
}

std::map<std::string, uint32_t> GetTensorMap(
    std::vector<std::string> &inTensorList, std::vector<std::string> &outTensorList,
    std::vector<std::string> &intermediateTensorList)
{
    std::map<std::string, uint32_t> tensorMap = {};
    uint32_t tensorIdx = 0;

    // translatedinTensor
    for (const auto &tensor : inTensorList) {
        tensorMap[tensor] = tensorIdx;
        tensorIdx++;
    }

    // translatedoutTensor
    for (const auto &tensor : outTensorList) {
        tensorMap[tensor] = tensorIdx;
        tensorIdx++;
    }

    // translatedintermediateTensor
    for (const auto &tensor : intermediateTensorList) {
        tensorMap[tensor] = tensorIdx;
        tensorIdx++;
    }

    std::stringstream ss;
    for (auto tensor = tensorMap.cbegin(); tensor != tensorMap.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("tensor map\n" << ss.str());

    return tensorMap;
}

uint32_t GetTensorIdx(const std::map<std::string, uint32_t> &tensorMap, std::string tensorName)
{
    if (tensorMap.find(tensorName) == tensorMap.end()) {
        ATB_SPEED_LOG_DEBUG("Cannot find " << tensorName << " in tensor Map");
        return UINT32_MAX;
    }
    return tensorMap.at(tensorName);
}

atb::SVector<uint32_t> GetTensorIdxList(const std::map<std::string, uint32_t> &tensorMap,
    std::vector<std::string>tensorNames)
{
    atb::SVector<uint32_t> tensorIdxList = {};
    for (std::string tensorName : tensorNames) {
        tensorIdxList.push_back(GetTensorIdx(tensorMap, tensorName));
    }
    return tensorIdxList;
}

bool CheckAntiOutlier(const int &packQuantType)
{
    bool isAntiOutlier = packQuantType == atb_speed::common::MIX_W8A8_ANTI || \
        packQuantType == atb_speed::common::ALL_W8A8_ANTI || \
        packQuantType == atb_speed::common::ALL_W8A8SC_ANTI || \
        packQuantType == atb_speed::common::MIX_W8A8SC_ANTI || \
        packQuantType == atb_speed::common::ALL_W8A16_ANTI || \
        packQuantType == atb_speed::common::MIX_W8A16_ANTI || \
        packQuantType == atb_speed::common::ALL_W4A16_ANTI || \
        packQuantType == atb_speed::common::MIX_W4A16_ANTI || \
        packQuantType == atb_speed::common::ALL_W8A8_DYNAMIC_ANTI || \
        packQuantType == atb_speed::common::MIX_W8A8_DYNAMIC_ANTI || \
        packQuantType == atb_speed::common::ALL_W4A8_ANTI || \
        packQuantType == atb_speed::common::MIX_W4A8_ANTI;
    return isAntiOutlier;
}

bool CheckPack(const int &packQuantType, const std::vector<int> &linearDescs, const std::vector<int> &linearIndex)
{
    static const std::unordered_set<int> PackableQuantTypes = {
        atb_speed::common::ALL_FP,
        atb_speed::common::ALL_W8A16, atb_speed::common::ALL_W8A16_ANTI,
        atb_speed::common::ALL_W4A16, atb_speed::common::ALL_W4A16_ANTI,
        atb_speed::common::ALL_W8A8, atb_speed::common::ALL_W8A8_ANTI,
        atb_speed::common::ALL_W8A8SC, atb_speed::common::ALL_W8A8SC_ANTI,
        atb_speed::common::ALL_W8A8_DYNAMIC, atb_speed::common::ALL_W8A8_DYNAMIC_ANTI,
        atb_speed::common::ALL_W4A8, atb_speed::common::ALL_W4A8_ANTI
    };

    // "packable" packQuantType
    if (PackableQuantTypes.count(packQuantType) > 0) {
        return true;
    }
    // "unpackable" packQuantType
    if (packQuantType != atb_speed::common::PACK_QUANT_UNDEFINED) {
        return false;
    }
    // undefined packQuantType, check pack from linearDescs (assume the first desc to be valid)
    int currentDesc = LinearDesc::INVALID_DESC;
    for (const int &index : linearIndex) {
        if (index >= static_cast<int>(linearDescs.size())) {
            ATB_SPEED_LOG_WARN(index << " out of range in CheckPack");
            continue;
        }
        int desc = linearDescs.at(index);
        // skip invalid desc, usually placeholder for packed linear
        if (desc == LinearDesc::INVALID_DESC) {
            continue;
        }
        // init with first valid desc
        if (currentDesc == LinearDesc::INVALID_DESC) {
            currentDesc = desc;
        } else if (desc != currentDesc) {
            // if valid and differ from prev descs -> unpackable
            return false;
        }
    }
    return true;
}

atb::Status CheckParamVectorSize(const std::vector<int> &vector, size_t threshold)
{
    if (vector.size() < threshold) {
        return atb::ERROR_INVALID_PARAM;
    }
    return atb::NO_ERROR;
}

atb::Status CreateRecordWithoutNodeId(atb::GraphParam &opGraph,
                                      atb_speed::EventAction eventAction, const std::string &cvKey)
{
    atb::Node recordNode;
    recordNode.inTensorIds = {};
    recordNode.outTensorIds = {};
    CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().RecordEvent(
        recordNode.operation,
        eventAction,
        cvKey));
    opGraph.nodes.push_back(recordNode);
    ATB_SPEED_LOG_DEBUG("Record event success");
    return atb::NO_ERROR;
}

atb::Status CreateWaitWithoutNodeId(atb::GraphParam &opGraph,
                                    atb_speed::EventAction eventAction, const std::string &cvKey)
{
    atb::Node waitNode;
    waitNode.inTensorIds = {};
    waitNode.outTensorIds = {};
    CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().WaitEvent(
        waitNode.operation,
        eventAction,
        cvKey));
    opGraph.nodes.push_back(waitNode);
    ATB_SPEED_LOG_DEBUG("Wait event success");
    return atb::NO_ERROR;
}

PackQuantType ConvertQuantTypeToPackType(std::string quantType)
{
    const std::unordered_map<std::string, atb_speed::common::PackQuantType> quantTypeToPackType = {
        {"float", atb_speed::common::PackQuantType::ALL_FP},
        {"w8a8", atb_speed::common::PackQuantType::ALL_W8A8},
        {"w8a8s", atb_speed::common::PackQuantType::ALL_W8A8},
        {"w8a8sc", atb_speed::common::PackQuantType::ALL_W8A8SC},
        {"w8a8_dynamic", atb_speed::common::PackQuantType::ALL_W8A8_DYNAMIC},
        {"w8a16", atb_speed::common::PackQuantType::ALL_W8A16},
        {"w4a16", atb_speed::common::PackQuantType::ALL_W4A16},
        {"", atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED},
    };

    auto it = quantTypeToPackType.find(quantType);
    if (it == quantTypeToPackType.end()) {
        return atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    }

    return it->second;
}

template void AddTensorToList(
    const std::map<std::string, std::vector<std::string>> &tensorCandidates,
    std::string targetKey, std::vector<std::string> &tensorList);
template void AddTensorToList(
    const std::map<std::string, atb::SVector<std::string>> &tensorCandidates,
    std::string targetKey, atb::SVector<std::string> &tensorList);
} // namespace common
} // namespace atb_speed