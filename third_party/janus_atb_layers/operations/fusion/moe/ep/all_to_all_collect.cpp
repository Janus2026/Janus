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
 
#include "all_to_all_collect.h"
#include <atb/atb_infer.h>
#include <memory>
#include "operations/fusion/utils.h"

namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetAllToAllCollectInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> allToAllCollectInTensorCandidates = {
        {"default", {
            "in_hiddenstates", "in_moe_out", "in_mask", "in_shuffle_idx", "in_valid_idx"}
        },
    };
    return allToAllCollectInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetAllToAllCollectIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> allToAllCollectIntermediateTensorCandidates = {
        {"default", {
            "intermediate_recv_output", "intermediate_filtered_output"}
        },
        {"has_tp", {
            "intermediate_moe_output_partial", "intermediate_moe_output"}
        }
    };
    return allToAllCollectIntermediateTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetAllToAllCollectOutTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> allToAllCollectOutTensorCandidates = {
        {"default", {
            "out"}
        },
    };
    return allToAllCollectOutTensorCandidates;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const AllToAllCollectParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &interTensorNum)
{
    auto allToAllCollectInTensorCandidates = GetAllToAllCollectInTensorCandidates();
    auto allToAllCollectIntermediateTensorCandidates = GetAllToAllCollectIntermediateTensorCandidates();
    auto allToAllCollectOutTensorCandidates = GetAllToAllCollectOutTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> interTensorList = {};
    std::vector<std::string> outTensorList = {};

    AddTensorToList(allToAllCollectInTensorCandidates, "default", inTensorList);
    AddTensorToList(allToAllCollectIntermediateTensorCandidates, "default", interTensorList);
    if (param.hasMoeTp) {
        AddTensorToList(allToAllCollectIntermediateTensorCandidates, "has_tp", interTensorList);
    }
    AddTensorToList(allToAllCollectOutTensorCandidates, "default", outTensorList);

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    interTensorNum = interTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, interTensorList);
}

atb::Status CreateReduceScatterData(std::map<std::string, uint32_t> &tensorMap, const AllToAllCollectParam &param,
    atb::GraphParam &opGraph)
{
    atb::Node allToAllNode;
    atb::infer::AllToAllParam allToAllParam;
    allToAllParam.rank = param.mlpTpRank;
    allToAllParam.rankSize = param.mlpTpSize;
    allToAllParam.backend = param.backend;
    allToAllParam.hcclComm = param.hcclComm;
    allToAllParam.rankTableFile = param.mlpTpRankTableFile;
    allToAllParam.commDomain = param.mlpTpDomain;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(allToAllParam, &allToAllNode.operation));
    allToAllNode.inTensorIds = {GetTensorIdx(tensorMap, "in_moe_out")};
    allToAllNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_moe_output_partial")};
    opGraph.nodes.push_back(allToAllNode);

    atb::Node reduceNode;
    atb::infer::ReduceParam reduceParam;
    reduceParam.reduceType = atb::infer::ReduceParam::ReduceType::REDUCE_SUM;
    reduceParam.axis.resize(1); // translated SVector translated
    reduceParam.axis[0] = 0; // translated 1
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(reduceParam, &reduceNode.operation));
    reduceNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_moe_output_partial")};
    reduceNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_moe_output")};
    reduceNode.inTensorReshapeFuncs.resize(reduceNode.inTensorIds.size());
    reduceNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: dimNum
        newShape.dims[0] = param.mlpTpSize;
        newShape.dims[1] = oldShape.dims[0] / param.mlpTpSize;
        newShape.dims[2] = oldShape.dims[1]; // 2: dim 2
    };
    opGraph.nodes.push_back(reduceNode);
    return atb::NO_ERROR;
}

atb::Status CreateAll2AllCollectData(std::map<std::string, uint32_t> &tensorMap, const AllToAllCollectParam &param,
    atb::GraphParam &opGraph)
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
    if (param.hasMoeTp) {
        allToAllNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_moe_output")};
    } else {
        allToAllNode.inTensorIds = {GetTensorIdx(tensorMap, "in_moe_out")};
    }
    allToAllNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_recv_output")};
    opGraph.nodes.push_back(allToAllNode);
    return atb::NO_ERROR;
}

atb::Status CreateFilteredRecvData(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node node;
    atb::infer::ElewiseParam param;
    param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(param, &node.operation));
    node.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_recv_output"),
                        GetTensorIdx(tensorMap, "in_mask")};
    node.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_filtered_output")};
    node.inTensorReshapeFuncs.resize(node.inTensorIds.size());
    node.inTensorReshapeFuncs[1] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = 1;
    };
    opGraph.nodes.push_back(node);
    return atb::NO_ERROR;
}


atb::Status CreateEmptyTensor(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node mapNode;
    atb::infer::ElewiseParam mapParam;
    mapParam.mulsParam.varAttr = 0.0;
    mapParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(mapParam, &mapNode.operation));
    mapNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates")};
    mapNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(mapNode);
    return atb::NO_ERROR;
}


atb::Status CreateIndexAdd(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node indexAddNode;
    atb::infer::IndexAddParam indexAddParam;
    indexAddParam.indexType = atb::infer::IndexAddParam::IndexType::INDEX_ADD_VALID;
    indexAddParam.axis = 0;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(indexAddParam, &indexAddNode.operation));
    indexAddNode.inTensorIds = {GetTensorIdx(tensorMap, "out"),
                                GetTensorIdx(tensorMap, "in_shuffle_idx"),
                                GetTensorIdx(tensorMap, "intermediate_filtered_output"),
                                GetTensorIdx(tensorMap, "in_valid_idx")};
    indexAddNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(indexAddNode);
    return atb::NO_ERROR;
}

atb::Status CreateAllToAllCollectOperation(const AllToAllCollectParam &param, atb::Operation **operation)
{
    ATB_SPEED_LOG_DEBUG("CreateAllToAllCollectOperation Start");
    atb::GraphParam opGraph;
    opGraph.name = "AllToAllCollect";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);

    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    if (param.hasMoeTp) {
        CHECK_OPERATION_STATUS_RETURN(CreateReduceScatterData(tensorMap, param, opGraph));
    }
    CHECK_OPERATION_STATUS_RETURN(CreateAll2AllCollectData(tensorMap, param, opGraph));
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateFilteredRecvData(tensorMap, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateEmptyTensor(tensorMap, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateIndexAdd(tensorMap, opGraph));

    opGraph.inferShapeFunc = [=] (const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    ATB_SPEED_LOG_DEBUG("CreateAllToAllCollectOperation success");
    return atb::NO_ERROR;
}
}
} // namespace atb_speed