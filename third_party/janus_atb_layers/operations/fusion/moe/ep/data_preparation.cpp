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
#include "data_preparation.h"
#include <atb/atb_infer.h>
#include <memory>
#include "operations/fusion/utils.h"
#include "operations/aclnn/ops/moe_init_routing_operation.h"
#include "operations/aclnn/ops/moe_compute_expert_tokens_operation.h"

namespace atb_speed {
namespace common {
std::map<std::string, std::vector<std::string>> GetDataPreparationInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> dataPreparationInTensorCandidates = {
        {"default", {
            "in_selected_experts", "in_idx", "in_one_hot", "in_zero_hot"}},
    };
    return dataPreparationInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetDataPreparationInterTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> dataPreparationInterTensorCandidates = {
        {"default", {
            "intermediate_group_count"}},
    };
    return dataPreparationInterTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetDataPreparationOutTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> dataPreparationOutTensorCandidates = {
        {"default", {
            "out_shuffle_idx", "out_expert_idx", "out_group_count"}},
    };
    return dataPreparationOutTensorCandidates;
}

std::map<std::string, uint32_t> ConstructDataPreparationTensorMap(
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &interTensorNum)
{
    auto dataPreparationInTensorCandidates = GetDataPreparationInTensorCandidates();
    auto dataPreparationInterTensorCandidates = GetDataPreparationInterTensorCandidates();
    auto dataPreparationOutTensorCandidates = GetDataPreparationOutTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> interTensorList = {};
    std::vector<std::string> outTensorList = {};

    AddTensorToList(dataPreparationInTensorCandidates, "default", inTensorList);
    AddTensorToList(dataPreparationInterTensorCandidates, "default", interTensorList);
    AddTensorToList(dataPreparationOutTensorCandidates, "default", outTensorList);

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    interTensorNum = interTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, interTensorList);
}

atb::Status CreateDeviceGating(std::map<std::string, uint32_t> &tensorMap, const DataPreparationParam &param,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &gatingNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatingParam gatingParam;
    gatingParam.topkExpertNum = param.topk;
    gatingParam.cumSumNum = param.numOfExperts;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatingParam, &gatingNode.operation));

    gatingNode.inTensorIds = {GetTensorIdx(tensorMap, "in_selected_experts"),
                              GetTensorIdx(tensorMap, "in_idx")};
    gatingNode.outTensorIds = {GetTensorIdx(tensorMap, "out_shuffle_idx"),
                               GetTensorIdx(tensorMap, "intermediate_group_count"),
                               GetTensorIdx(tensorMap, "out_expert_idx")};

    gatingNode.inTensorReshapeFuncs.resize(gatingNode.inTensorIds.size());
    gatingNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1; // dimNum: 1
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };

    ATB_SPEED_LOG_DEBUG("Gating calculation success");
    return atb::NO_ERROR;
}


// translated Group Count translated,translatedworldsizetranslated, translated
atb::Status CreateGroupSlice(std::map<std::string, uint32_t> &tensorMap, const DataPreparationParam &param,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::infer::SliceParam sliceParam;
    atb::Node &sliceNode = opGraph.nodes.at(nodeId++);

    sliceParam.offsets.resize(2); // 2: dimNum
    sliceParam.offsets[0] = 0;
    sliceParam.offsets[1] = -1;

    sliceParam.size.resize(2); // 2: dimNum
    sliceParam.size[0] = param.worldSize;
    sliceParam.size[1] = 1;

    CreateOperation(sliceParam, &sliceNode.operation);

    sliceNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_count")};
    sliceNode.outTensorIds = {GetTensorIdx(tensorMap, "out_group_count")};

    sliceNode.inTensorReshapeFuncs.resize(sliceNode.inTensorIds.size());
    sliceNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: dimNum
        newShape.dims[0] = param.worldSize;
        newShape.dims[1] = oldShape.dims[0] / param.worldSize;
    };

    ATB_SPEED_LOG_DEBUG("CreateGroupSlice, Get Device Token Num");
    return atb::NO_ERROR;
}

atb::Status CreateDataPreparationOperation(const DataPreparationParam &param, atb::Operation **operation)
{
    ATB_SPEED_LOG_DEBUG("CreateDataPreparationOperation Start");
    atb::GraphParam opGraph;
    opGraph.name = "DataPreparation";
    std::map<std::string, uint32_t> tensorMap = ConstructDataPreparationTensorMap(
        opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);

    uint64_t nodeCount = 2;
    opGraph.nodes.resize(nodeCount);
    size_t nodeId = 0;

    CHECK_OPERATION_STATUS_RETURN(CreateDeviceGating(tensorMap, param, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateGroupSlice(tensorMap, param, nodeId, opGraph));

    opGraph.inferShapeFunc = [=] (const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        size_t shape = inTensorDescs.at(1).shape.dims[0];
        outTensorDescs.at(0) = inTensorDescs.at(1);
        outTensorDescs.at(0).shape.dimNum = 1;
        outTensorDescs.at(0).shape.dims[0] = shape;
        outTensorDescs.at(1) = inTensorDescs.at(1);
        outTensorDescs.at(1).shape.dimNum = 1;
        outTensorDescs.at(1).shape.dims[0] = shape;
        outTensorDescs.at(2) = inTensorDescs.at(1); // 2: dim 2
        outTensorDescs.at(2).shape.dimNum = 1; // 2: dim 2
        outTensorDescs.at(2).shape.dims[0] = param.worldSize; // 2: dim 2
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    ATB_SPEED_LOG_DEBUG("CreateDataPreparationOperation success");
    return atb::NO_ERROR;
}
}
} // namespace atb_speed