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
#include "all_to_all_dispatch.h"
#include <atb/atb_infer.h>
#include <memory>
#include "operations/fusion/utils.h"
namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetAllToAllDispatchInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> allToAllDispatchInTensorCandidates = {
        {"default", {
            "in_hiddenstatus", "in_selected_experts", "in_expert_weight",
            "in_hidden_shuffle_idx", "in_expert_shuffle_idx",
            "in_zero_hot", "in_one_hot"}
        },
    };
    return allToAllDispatchInTensorCandidates;
}


std::map<std::string, std::vector<std::string>> GetAllToAllDispatchInterTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> allToAllDispatchInterTensorCandidates = {
        {"default", {
            "intermediate_send_expert", "intermediate_send_hiddenstatus"}
        },
    };
    return allToAllDispatchInterTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetAllToAllDispatchOutTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> allToAllDispatchOutTensorCandidates = {
        {"default", {
            "out_hiddenstates", "out_selected_experts", "out_expert_weight"}
        },
    };
    return allToAllDispatchOutTensorCandidates;
}

std::map<std::string, uint32_t> ConstructAllToAllDispatchTensorMap(
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &interTensorNum)
{
    auto allToAllDispatchInTensorCandidates = GetAllToAllDispatchInTensorCandidates();
    auto allToAllDispatchInterTensorCandidates = GetAllToAllDispatchInterTensorCandidates();
    auto allToAllDispatchOutTensorCandidates = GetAllToAllDispatchOutTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> interTensorList = {};
    std::vector<std::string> outTensorList = {};

    AddTensorToList(allToAllDispatchInTensorCandidates, "default", inTensorList);
    AddTensorToList(allToAllDispatchInterTensorCandidates, "default", interTensorList);
    AddTensorToList(allToAllDispatchOutTensorCandidates, "default", outTensorList);
    
    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    interTensorNum = interTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, interTensorList);
}

atb::Status CreateExpertGather(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
    gatherNode.inTensorIds = {GetTensorIdx(tensorMap, "in_selected_experts"),
                              GetTensorIdx(tensorMap, "in_expert_shuffle_idx")};
    gatherNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_send_expert")};
    gatherNode.inTensorReshapeFuncs.resize(gatherNode.inTensorIds.size());
    gatherNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    opGraph.nodes.push_back(gatherNode);
    return atb::NO_ERROR;
}

atb::Status CreateExpertWeightGather(std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph)
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
    gatherNode.inTensorIds = {GetTensorIdx(tensorMap, "in_expert_weight"),
                              GetTensorIdx(tensorMap, "in_expert_shuffle_idx")};
    gatherNode.outTensorIds = {GetTensorIdx(tensorMap, "out_expert_weight")};
    gatherNode.inTensorReshapeFuncs.resize(gatherNode.inTensorIds.size());
    gatherNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    opGraph.nodes.push_back(gatherNode);
    return atb::NO_ERROR;
}

atb::Status CreateHiddenGather(std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph)
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
    gatherNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstatus"),
                              GetTensorIdx(tensorMap, "in_hidden_shuffle_idx")};
    gatherNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_send_hiddenstatus")};
    opGraph.nodes.push_back(gatherNode);
    return atb::NO_ERROR;
}

atb::Status CreateAll2AllDispatchData(std::map<std::string, uint32_t> &tensorMap,
    const AllToAllDispatchParam &param, atb::GraphParam &opGraph)
{
    atb::Node allToAllNode;
    atb::infer::AllToAllParam allToAllParam;
    allToAllParam.rank = param.moeEpRank;
    allToAllParam.rankSize = param.moeEpSize;
    allToAllParam.backend = param.backend;
    allToAllParam.hcclComm = param.hcclComm;
    allToAllParam.rankTableFile = param.moeEpRankTableFile;
    allToAllParam.commDomain=param.moeEpDomain;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(allToAllParam, &allToAllNode.operation));
    allToAllNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_send_hiddenstatus")};
    allToAllNode.outTensorIds = {GetTensorIdx(tensorMap, "out_hiddenstates")};
    opGraph.nodes.push_back(allToAllNode);
    return atb::NO_ERROR;
}

atb::Status CreateAll2AllDispatchExpert(std::map<std::string, uint32_t> &tensorMap,
    const AllToAllDispatchParam &param, atb::GraphParam &opGraph)
{
    atb::Node allToAllNode;
    atb::infer::AllToAllParam allToAllParam;
    allToAllParam.rank = param.moeEpRank;
    allToAllParam.rankSize = param.moeEpSize;
    allToAllParam.backend = param.backend;
    allToAllParam.hcclComm = param.hcclComm;
    allToAllParam.rankTableFile = param.moeEpRankTableFile;
    allToAllParam.commDomain = param.moeEpDomain;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(allToAllParam, &allToAllNode.operation));
    allToAllNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_send_expert")};
    allToAllNode.outTensorIds = {GetTensorIdx(tensorMap, "out_selected_experts")};
    opGraph.nodes.push_back(allToAllNode);
    return atb::NO_ERROR;
}

atb::Status CreateAllToAllDispatchOperation(const AllToAllDispatchParam &param, atb::Operation **operation)
{
    ATB_SPEED_LOG_DEBUG("CreateAllToAllDispatchOperation Start");
    atb::GraphParam opGraph;
    opGraph.name = "AllToAllDispatch";
    std::map<std::string, uint32_t> tensorMap = ConstructAllToAllDispatchTensorMap(
        opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);

    CHECK_OPERATION_STATUS_RETURN(CreateExpertGather(tensorMap, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateExpertWeightGather(tensorMap, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateHiddenGather(tensorMap, opGraph));
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateAll2AllDispatchData(tensorMap, param, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateAll2AllDispatchExpert(tensorMap, param, opGraph));
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));

    opGraph.inferShapeFunc = [=] (const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(1) = inTensorDescs.at(1);
        outTensorDescs.at(2) = inTensorDescs.at(2); /// 2: dim 2

        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(3).shape.dims[0]; // 3: dim 3
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(3).shape.dims[0]; // 2: dim2, 3: dim 3
        outTensorDescs.at(2).shape.dims[0] = inTensorDescs.at(3).shape.dims[0]; // 2: dim2, 3: dim 3

        outTensorDescs.at(1).shape.dims[1] = 1;
        outTensorDescs.at(2).shape.dims[1] = 1; // 2: dim 2
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
}
} //