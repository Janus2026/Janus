#pragma once
#include <vector>
#include <string>
#include "atb/atb_infer.h"

namespace janus {
namespace atb_utils {

inline bool inited=false;

inline void append(std::vector<std::string>& target, 
    const std::vector<std::string>& to_append) {
    target.insert(target.end(), to_append.begin(), to_append.end());
    return;
}

inline atb::Status CreateNewStreamRecordWithoutNodeId(atb::GraphParam &opGraph, atb_speed::EventAction eventAction,
    const std::string &cvKey)
{
    atb::Node recordNode;
    recordNode.inTensorIds = {};
    recordNode.outTensorIds = {};
    CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().RecordEvent(
        recordNode.operation,
        eventAction,
        cvKey));
    atb::SetExecuteStreamId(recordNode.operation, 1);
    opGraph.nodes.push_back(recordNode);
    ATB_SPEED_LOG_DEBUG("Record event success");
    return atb::NO_ERROR;
}

inline atb::Status CreateNewStreamWaitWithoutNodeId(atb::GraphParam &opGraph, atb_speed::EventAction eventAction,
    const std::string &cvKey)
{
    atb::Node waitNode;
    waitNode.inTensorIds = {};
    waitNode.outTensorIds = {};
    CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().WaitEvent(
        waitNode.operation,
        eventAction,
        cvKey));
    atb::SetExecuteStreamId(waitNode.operation, 1);
    opGraph.nodes.push_back(waitNode);
    ATB_SPEED_LOG_DEBUG("Wait event success");
    return atb::NO_ERROR;
}

inline atb::Status insert_start_events(atb::GraphParam &opGraph) {
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateRecordWithoutNodeId(
        opGraph, atb_speed::EventAction::PUSH, atb_speed::common::CC_START));
    CHECK_OPERATION_STATUS_RETURN(CreateNewStreamWaitWithoutNodeId(
        opGraph, atb_speed::EventAction::POP, atb_speed::common::CC_START));
    return atb::NO_ERROR;
}

inline atb::Status insert_push_events(atb::GraphParam &opGraph) {
    CHECK_OPERATION_STATUS_RETURN(CreateNewStreamRecordWithoutNodeId(
        opGraph, atb_speed::EventAction::PUSH, atb_speed::common::COMM_CONTROL));
    CHECK_OPERATION_STATUS_RETURN(CreateNewStreamWaitWithoutNodeId(
       opGraph, atb_speed::EventAction::PUSH, atb_speed::common::COMP_CONTROL));
    inited = true;
    return atb::NO_ERROR;
}

inline atb::Status insert_pop_events(atb::GraphParam &opGraph) {
    if (!inited) {
        insert_start_events(opGraph);
    } else {
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateRecordWithoutNodeId(
            opGraph, atb_speed::EventAction::POP, atb_speed::common::COMP_CONTROL));
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateWaitWithoutNodeId(
            opGraph, atb_speed::EventAction::POP, atb_speed::common::COMM_CONTROL));
    }
    return atb::NO_ERROR;
}

inline atb::Status insert_end_events(atb::GraphParam &opGraph) {
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateRecordWithoutNodeId(
        opGraph, atb_speed::EventAction::PUSH, atb_speed::common::END_EVENT));
    CHECK_OPERATION_STATUS_RETURN(CreateNewStreamWaitWithoutNodeId(
        opGraph, atb_speed::EventAction::POP, atb_speed::common::END_EVENT));
    return atb::NO_ERROR;
}

} // namespace atb_utils
} // namespace janus