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
#include "expert_filter.h"
#include <atb/atb_infer.h>
#include <memory>
#include "operations/fusion/utils.h"
#include "operations/aclnn/ops/inplacemasked_filltensor_operation.h"


namespace atb_speed {
namespace common {
std::map<std::string, std::vector<std::string>> GetExpertFilterInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> expertFilterInTensorCandidates = {
        {"default", {
            "in_selected_experts", "in_expert_weight", "in_start_expert_idx",
            "in_device_expert_count", "in_zero_hot"}
        },
    };
    return expertFilterInTensorCandidates;
}


std::map<std::string, std::vector<std::string>> GetExpertFilterInterTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> expertFilterInterTensorCandidates = {
        {"default", {
            "intermediate_selected_experts_int64", "intermediate_selected_experts_mask",
            "intermediate_selected_experts_shifted_int32", "intermediate_selected_experts_shifted_int64",
            "intermediate_selected_experts_mask_1", "intermediate_zero_hot_int64"}
        },
        {"shifted", {
            "intermediate_selected_experts_int64", "intermediate_selected_experts_mask_1"}
        },
    };
    return expertFilterInterTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetExpertFilterOutTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> expertFilterOutTensorCandidates = {
        {"default", {
            "out_selected_experts", "out_expert_weight"}
        },
        {"shifted", {
            "out_expert_weight"}
        },
    };
    return expertFilterOutTensorCandidates;
}

std::map<std::string, uint32_t> ConstructTensorMap(const ExpertFilterParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &interTensorNum)
{
    auto expertFilterInTensorCandidates = GetExpertFilterInTensorCandidates();
    auto expertFilterInterTensorCandidates = GetExpertFilterInterTensorCandidates();
    auto expertFilterOutTensorCandidates = GetExpertFilterOutTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> interTensorList = {};
    std::vector<std::string> outTensorList = {};

    AddTensorToList(expertFilterInTensorCandidates, "default", inTensorList);
    if (param.shiftedTopK && !param.enableGatingDp) {
        AddTensorToList(expertFilterInterTensorCandidates, "shifted", interTensorList);
        AddTensorToList(expertFilterOutTensorCandidates, "shifted", outTensorList);
    } else {
        AddTensorToList(expertFilterInterTensorCandidates, "default", interTensorList);
        AddTensorToList(expertFilterOutTensorCandidates, "default", outTensorList);
    }
    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    interTensorNum = interTensorList.size();
    return GetTensorMap(inTensorList, outTensorList, interTensorList);
}

atb::Status CreateSelectedExpertInt64(std::map<std::string, uint32_t> &tensorMap,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &castNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_INT64;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {GetTensorIdx(tensorMap, "in_selected_experts")};
    castNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_selected_experts_int64")};
    return atb::NO_ERROR;
}

atb::Status CreateSelectedExpertSub(std::map<std::string, uint32_t> &tensorMap,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &subNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam subParam;
    subParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_SUB;
    CreateOperation(subParam, &subNode.operation);
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(subParam, &subNode.operation));
    subNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_selected_experts_int64"),
                           GetTensorIdx(tensorMap, "in_start_expert_idx")};
    subNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_selected_experts_shifted_int64")};

    return atb::NO_ERROR;
}

atb::Status CreateZeroHotInt64(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &castNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_INT64;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {GetTensorIdx(tensorMap, "in_zero_hot")};
    castNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_zero_hot_int64")};
    return atb::NO_ERROR;
}

atb::Status CreateSelectedExpertMask(std::map<std::string, uint32_t> &tensorMap,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &node = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam lessParam;
    lessParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_LESS;
    CreateOperation(lessParam, &node.operation);
    node.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_selected_experts_shifted_int64"),
                        GetTensorIdx(tensorMap, "intermediate_zero_hot_int64")};
    node.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_selected_experts_mask")};
    return atb::NO_ERROR;
}

atb::Status CreateSelectedExpertInt32(std::map<std::string, uint32_t> &tensorMap,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &castNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_INT32;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_selected_experts_shifted_int64")};
    castNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_selected_experts_shifted_int32")};
    return atb::NO_ERROR;
}

atb::Status CreateOutSelectedExpert(std::map<std::string, uint32_t> &tensorMap,
    const ExpertFilterParam &param, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &node = opGraph.nodes.at(nodeId++);
    atb::infer::FillParam fillParam;
    fillParam.withMask = true;
    fillParam.value.resize(1);
    fillParam.value[0] = param.numOfExperts;
    CreateOperation(fillParam, &node.operation);
    node.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_selected_experts_shifted_int32"),
                        GetTensorIdx(tensorMap, "intermediate_selected_experts_mask")};
    node.outTensorIds = {GetTensorIdx(tensorMap, "out_selected_experts")};
    return atb::NO_ERROR;
}

atb::Status CreateSelectedExpertMask1(
    std::map<std::string, uint32_t> &tensorMap, const ExpertFilterParam &param, size_t &nodeId,
    atb::GraphParam &opGraph)
{
    atb::Node &node = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam lessParam;
    lessParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_GREATER;
    CreateOperation(lessParam, &node.operation);
    node.inTensorIds = {
        GetTensorIdx(tensorMap, (param.shiftedTopK && !param.enableGatingDp) ? \
            "intermediate_selected_experts_int64" : "intermediate_selected_experts_shifted_int64"),
        GetTensorIdx(tensorMap, "in_device_expert_count")};
    node.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_selected_experts_mask_1")};
    return atb::NO_ERROR;
}

atb::Status CreateExpertWeightFilter(
    std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &node = opGraph.nodes.at(nodeId++);
    atb::infer::FillParam param;
    param.withMask = true;
    param.value.resize(1);
    param.value[0] = 0;

    CreateOperation(param, &node.operation);

    node.inTensorIds = {GetTensorIdx(tensorMap, "in_expert_weight"),
                        GetTensorIdx(tensorMap, "intermediate_selected_experts_mask")};
    node.outTensorIds = {GetTensorIdx(tensorMap, "out_expert_weight")};

    return atb::NO_ERROR;
}

atb::Status CreateOutExpertWeightFilter(
    std::map<std::string, uint32_t> &tensorMap, const ExpertFilterParam &param, size_t &nodeId,
    atb::GraphParam &opGraph)
{
    atb::Node &node = opGraph.nodes.at(nodeId++);
    atb_speed::common::InplaceMaskedFillTensorParam inplaceMaskedFillTensorParam;
    inplaceMaskedFillTensorParam.value = 0;
    inplaceMaskedFillTensorParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    node.operation = new atb_speed::common::InplaceMaskedFillTensorOperation("MaskFillNode",
                                inplaceMaskedFillTensorParam);
    node.inTensorIds = {GetTensorIdx(tensorMap, (param.shiftedTopK && !param.enableGatingDp) ? \
                                    "in_expert_weight" : "out_expert_weight"),
                        GetTensorIdx(tensorMap, "intermediate_selected_experts_mask_1")};
    node.outTensorIds = {GetTensorIdx(tensorMap, "out_expert_weight")};

    return atb::NO_ERROR;
}

atb::Status CreateExpertFilterOperation(const ExpertFilterParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "ExpertFilter";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(param,
        opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    uint64_t nodeCount = 9; // 9: default node count
    if (param.shiftedTopK && !param.enableGatingDp) {
        nodeCount = 3; // 3: node count
    }
    opGraph.nodes.resize(nodeCount);
    size_t nodeId = 0;
    if (!param.shiftedTopK || param.enableGatingDp) {
        CHECK_OPERATION_STATUS_RETURN(CreateSelectedExpertInt64(tensorMap, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateSelectedExpertSub(tensorMap, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateZeroHotInt64(tensorMap, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateSelectedExpertMask(tensorMap, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateSelectedExpertInt32(tensorMap, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateOutSelectedExpert(tensorMap, param, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateSelectedExpertMask1(tensorMap, param, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateExpertWeightFilter(tensorMap, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateOutExpertWeightFilter(tensorMap, param, nodeId, opGraph));
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateSelectedExpertInt64(tensorMap, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateSelectedExpertMask1(tensorMap, param, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateOutExpertWeightFilter(tensorMap, param, nodeId, opGraph));
    }
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        if (param.shiftedTopK && !param.enableGatingDp) {
            outTensorDescs.at(0) = inTensorDescs.at(1);
        } else {
            outTensorDescs.at(0) = inTensorDescs.at(0);
            outTensorDescs.at(1) = inTensorDescs.at(1);
        }
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
}
} // namespace atb_speed