/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#include "dynamic_ep_moe.h"
#include <atb/atb_infer.h>
#include <memory>
#include "operations/aclnn/ops/argsort_operation.h"
#include "operations/aclnn/ops/grouped_matmul_operation.h"
#include "operations/fusion/moe/moe_mlp.h"
#include "data_preparation.h"
#include "all_to_all_meta.h"
#include "all_to_all_dispatch.h"
#include "all_to_all_collect.h"
#include "operations/fusion/utils.h"
#include "fused_alltoall_gmm.h"


namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetDynamicEpMoEInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> dynamicEpMoEInTensorCandidates = {
        {"default", {
            "in_hiddenstatus", "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert",
            "in_mlp_gateup_descale_expert", "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert",
            "in_mlp_gateup_compress_idx_expert", "in_mlp_down_weight_expert",
            "in_mlp_down_bias_expert", "in_mlp_down_descale_expert", "in_mlp_down_offset_expert",
            "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert", "in_expert_array", "in_selected_experts",
            "in_expert_weight", "in_one_hot", "in_zero_hot"}
        },
        {"ep", {
            "in_start_expert_idx", "in_device_expert_count", "in_padding_idx"}
        },
        {"dynamic_ep", {
            "in_buffer_idx", "in_moe_idx"}
        },
    };
    return dynamicEpMoEInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetDynamicEpMoEInterTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> dynamicEpMoEInterTensorCandidates = {
        {"default", {
            "intermediate_shuffle_idx", "intermediate_expert_shuffle_idx", "intermediate_valid_idx",
            "intermediate_buffer_idx", "intermediate_shuffle_idx_1",
            "intermediate_shuffle_idx_2", "intermediate_expert_shuffle_idx_1",
            "intermediate_group_count", "intermediate_shuffle_weight", "intermediate_recv_hiddenstatus",
            "intermediate_recv_selected_experts", "intermediate_experts_weight", "intermediate_moe_output"}
        },
    };
    return dynamicEpMoEInterTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetDynamicEpMoEOutTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> dynamicEpMoEOutTensorCandidates = {
        {"default", {
            "out_hiddenstates"}
        },
    };
    return dynamicEpMoEOutTensorCandidates;
}

std::map<std::string, uint32_t> ConstructDynamicEpTensorMap(
    const DynamicEpMoEParam &param, uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &interTensorNum)
{
    auto dynamicEpMoEInTensorCandidates = GetDynamicEpMoEInTensorCandidates();
    auto dynamicEpMoEInterTensorCandidates = GetDynamicEpMoEInterTensorCandidates();
    auto dynamicEpMoEOutTensorCandidates = GetDynamicEpMoEOutTensorCandidates();
    std::vector<std::string> inTensorList = {};
    std::vector<std::string> interTensorList = {};
    std::vector<std::string> outTensorList = {};
    AddTensorToList(dynamicEpMoEInTensorCandidates, "default", inTensorList);
    if (param.hasMoeEp) {
        AddTensorToList(dynamicEpMoEInTensorCandidates, "ep", inTensorList);
        if (param.isDynamicEp) {
            AddTensorToList(dynamicEpMoEInTensorCandidates, "dynamic_ep", inTensorList);
        }
    }
    if (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute && !param.enableLcocAll2All) {
        AddTensorToList(dynamicEpMoEInterTensorCandidates, "default", interTensorList);
    }
    AddTensorToList(dynamicEpMoEOutTensorCandidates, "default", outTensorList);
    if (param.enableExpertCumSumOutput) {
        outTensorList.push_back("out_gmm_cumsum_list");
    }

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    interTensorNum = interTensorList.size();
    return GetTensorMap(inTensorList, outTensorList, interTensorList);
}

atb::Status CreateFusedAllToAllMlp(std::map<std::string, uint32_t> &tensorMap, const DynamicEpMoEParam &param,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    auto &expertNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::All2AllMatmulParam mlpExpertParam;
    mlpExpertParam.topk = param.topk;
    mlpExpertParam.scaledTopk = param.scaledTopk;
    mlpExpertParam.numOfDeviceExperts = param.numOfDeviceExperts;
    mlpExpertParam.numOfExperts = param.numOfExperts;
    mlpExpertParam.gateUpTransposeB = param.gateUpTransposeB;
    mlpExpertParam.downTransposeB = param.downTransposeB;
    mlpExpertParam.moeEpRank = param.moeEpRank;
    mlpExpertParam.moeEpSize = param.moeEpSize;
    mlpExpertParam.lcclMoeEpDomain = param.lcclMoeEpDomain;
    mlpExpertParam.lcclMoeEpHcclComm = param.lcclMoeEpHcclComm;
    atb_speed::common::CreateAll2AllMatmulOperation(mlpExpertParam, &expertNode.operation);
 
    expertNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_hiddenstatus"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_weight_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_descale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_offset_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_scale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_compress_idx_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_weight_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_descale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_offset_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_scale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_compress_idx_expert"),
        GetTensorIdx(tensorMap, "in_expert_array"),
        GetTensorIdx(tensorMap, "in_selected_experts"),
        GetTensorIdx(tensorMap, "in_expert_weight"),
        GetTensorIdx(tensorMap, "in_moe_idx"),
    };
    expertNode.outTensorIds = {GetTensorIdx(tensorMap, "out_hiddenstates")};
 
    ATB_SPEED_LOG_DEBUG("Expert Group calculation success");
    return atb::NO_ERROR;
}
 

atb::Status CreateDataPreparation(std::map<std::string, uint32_t> &tensorMap,
    const DynamicEpMoEParam &param, size_t &nodeId, atb::GraphParam &opGraph)
{
    auto &dataPreparationNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::DataPreparationParam dataPreparationParam;
    dataPreparationParam.topk = param.topk;
    dataPreparationParam.numOfExperts = param.numOfExperts;
    dataPreparationParam.rank = param.moeEpRank;
    dataPreparationParam.worldSize = param.moeEpSize;
    dataPreparationParam.mixSharedRouting = param.mixSharedRouting;
    atb_speed::common::CreateDataPreparationOperation(dataPreparationParam, &dataPreparationNode.operation);
    dataPreparationNode.inTensorIds = {GetTensorIdx(tensorMap, "in_selected_experts"),
                                       GetTensorIdx(tensorMap, "in_buffer_idx"),
                                       GetTensorIdx(tensorMap, "in_one_hot"),
                                       GetTensorIdx(tensorMap, "in_zero_hot")};
    dataPreparationNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx"),
                                        GetTensorIdx(tensorMap, "intermediate_expert_shuffle_idx"),
                                        GetTensorIdx(tensorMap, "intermediate_group_count")};
    dataPreparationNode.inTensorChunks.resize(dataPreparationNode.inTensorIds.size());
    ATB_SPEED_LOG_DEBUG("dataPreparationNode calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateAllToAllMeta(std::map<std::string, uint32_t> &tensorMap, const DynamicEpMoEParam &param,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    auto &allToAllMetaNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::AllToAllMetaParam allToAllMetaParam;
    allToAllMetaParam.topk = param.topk;
    allToAllMetaParam.numOfExperts = param.numOfExperts;
    allToAllMetaParam.rank = param.moeEpRank;
    allToAllMetaParam.worldSize = param.moeEpSize;
    allToAllMetaParam.isBF16 = param.isBF16;
    atb_speed::common::CreateAllToAllMetaOperation(allToAllMetaParam, &allToAllMetaNode.operation);
    allToAllMetaNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_count"),
                                    GetTensorIdx(tensorMap, "in_moe_idx"),
                                    GetTensorIdx(tensorMap, "in_zero_hot")};
    allToAllMetaNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_buffer_idx"),
                                     GetTensorIdx(tensorMap, "intermediate_shuffle_weight"),
                                     GetTensorIdx(tensorMap, "intermediate_valid_idx")};
    allToAllMetaNode.inTensorChunks.resize(allToAllMetaNode.inTensorIds.size());
    ATB_SPEED_LOG_DEBUG("allToAllMetaNode calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateShuffleIdxDE(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &gatherNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
    gatherNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx"),
                              GetTensorIdx(tensorMap, "intermediate_buffer_idx")};
    gatherNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx_1")};
    ATB_SPEED_LOG_DEBUG("CreateShuffleIdxDE");
    return atb::NO_ERROR;
}

atb::Status CreateExpertShuffleIdx(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &gatherNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
    gatherNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_expert_shuffle_idx"),
                              GetTensorIdx(tensorMap, "intermediate_buffer_idx")};
    gatherNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_expert_shuffle_idx_1")};
    ATB_SPEED_LOG_DEBUG("CreateExpertShuffleIdx");
    return atb::NO_ERROR;
}

atb::Status CreateAllToAllDispatch(std::map<std::string, uint32_t> &tensorMap, const DynamicEpMoEParam &param,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    auto &allToAllDispatchNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::AllToAllDispatchParam allToAllDispatchParam;

    allToAllDispatchParam.topk = param.topk;
    allToAllDispatchParam.numOfExperts = param.numOfExperts;
    allToAllDispatchParam.backend = param.backend;
    allToAllDispatchParam.hcclComm = param.hcclComm;
    allToAllDispatchParam.hasMoeEp = param.hasMoeEp;
    allToAllDispatchParam.moeEpRank = param.moeEpRank;
    allToAllDispatchParam.moeEpSize = param.moeEpSize;
    allToAllDispatchParam.moeEpRankTableFile = param.moeEpRankTableFile;
    allToAllDispatchParam.moeEpDomain = param.moeEpDomain;

    allToAllDispatchParam.hasMlpTp = param.hasMlpTp;
    allToAllDispatchParam.mlpTpRank = param.mlpTpRank;
    allToAllDispatchParam.mlpTpSize = param.mlpTpSize;
    allToAllDispatchParam.mlpTpRankTableFile = param.mlpTpRankTableFile;
    allToAllDispatchParam.mlpTpDomain = param.mlpTpDomain;

    atb_speed::common::CreateAllToAllDispatchOperation(allToAllDispatchParam, &allToAllDispatchNode.operation);
    allToAllDispatchNode.inTensorIds = {
                                GetTensorIdx(tensorMap, "in_hiddenstatus"),
                                GetTensorIdx(tensorMap, "in_selected_experts"),
                                GetTensorIdx(tensorMap, "in_expert_weight"),
                                GetTensorIdx(tensorMap, "intermediate_shuffle_idx_1"),
                                GetTensorIdx(tensorMap, "intermediate_expert_shuffle_idx_1"),
                                GetTensorIdx(tensorMap, "in_zero_hot"),
                                GetTensorIdx(tensorMap, "in_one_hot"),
                                };

    allToAllDispatchNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_recv_hiddenstatus"),
                                GetTensorIdx(tensorMap, "intermediate_recv_selected_experts"),
                                GetTensorIdx(tensorMap, "intermediate_experts_weight"),
                                };

    allToAllDispatchNode.inTensorChunks.resize(allToAllDispatchNode.inTensorIds.size());
    ATB_SPEED_LOG_DEBUG("allToAllDispatchNode calculation success");
    return atb::NO_ERROR;
}

atb::Status SetMoeMlpParam(atb_speed::common::MoeMlpParam &mlpExpertParam, const DynamicEpMoEParam &param)
{
    if (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute) {
        mlpExpertParam.topk = 1;
    } else {
        mlpExpertParam.topk = param.topk;
        mlpExpertParam.scaledTopk = param.scaledTopk;
        mlpExpertParam.enableInitRoutingCutoff = param.enableInitRoutingCutoff;
    }
    mlpExpertParam.numOfDeviceExperts = param.numOfDeviceExperts;
    mlpExpertParam.hasMoeEp = param.hasMoeEp;
    mlpExpertParam.deviceExpert = param.deviceExpert;
    mlpExpertParam.expertParallelDegree = param.expertParallelDegree;
    mlpExpertParam.transpose = param.transpose;
    mlpExpertParam.numOfExperts = param.numOfExperts;
    mlpExpertParam.supportSwiGLU = param.supportSwiGLU;
    mlpExpertParam.moeLinearQuantType = param.moeLinearQuantType;
    mlpExpertParam.packQuantType = param.packQuantType;
    mlpExpertParam.denseQuantType = param.denseQuantType;
    mlpExpertParam.isBF16 = param.isBF16;
    mlpExpertParam.gateUpTransposeB = param.gateUpTransposeB;
    mlpExpertParam.downTransposeB = param.downTransposeB;
    mlpExpertParam.enableFusedRouting = param.enableFusedRouting;
    mlpExpertParam.enableInitQuant = param.enableInitQuant;
    mlpExpertParam.enableInitRoutingV3 = param.enableInitRoutingV3;
    mlpExpertParam.enableSwigluQuant = param.enableSwigluQuant;
    mlpExpertParam.enableAtlasGMMFused = param.enableAtlasGMMFused;
    mlpExpertParam.quantGroupSize = param.quantGroupSize;
    mlpExpertParam.enableGMMSwigluQuant = param.enableGMMSwigluQuant;
    mlpExpertParam.enableCVOverlap = param.enableCVOverlap;
    mlpExpertParam.backend = param.backend;
    mlpExpertParam.hasMoeEp = param.hasMoeEp;
    mlpExpertParam.moeEpRank = param.moeEpRank;
    mlpExpertParam.moeEpSize = param.moeEpSize;
    mlpExpertParam.moeEpDomain = param.moeEpDomain;
    mlpExpertParam.maxDecodeDpTokenSize = param.maxDecodeDpTokenSize;
    if (param.expertParallelDegree == 1) {
        mlpExpertParam.shiftedTopK = true;
    }
    mlpExpertParam.enableMoeDistribute = param.enableMoeDistribute && param.isDynamicEp;
    mlpExpertParam.enableExpertCumSumOutput = param.enableExpertCumSumOutput;
    mlpExpertParam.enableGatingDp = param.enableGatingDp;
    mlpExpertParam.enableDispatchCombineV2 = param.enableDispatchCombineV2;
    mlpExpertParam.numDanglingSharedExperts = param.numDanglingSharedExperts;
    mlpExpertParam.numOfRedundantExpert = param.numOfRedundantExpert;
    mlpExpertParam.enableIndexGmm = param.enableIndexGmm;
    mlpExpertParam.swigluBackend = param.swigluBackend;
    return atb::NO_ERROR;
}

atb::Status CreateMoeMlp(std::map<std::string, uint32_t> &tensorMap, const DynamicEpMoEParam &param,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    auto &expertNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::MoeMlpParam mlpExpertParam;
    SetMoeMlpParam(mlpExpertParam, param);
    atb_speed::common::CreateMoeMlpOperation(mlpExpertParam, &expertNode.operation);
    expertNode.outTensorIds = {GetTensorIdx(tensorMap,
        (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute) ? \
        "intermediate_moe_output" : "out_hiddenstates")};
    if (param.enableExpertCumSumOutput) {
        expertNode.outTensorIds.push_back(GetTensorIdx(tensorMap, "out_gmm_cumsum_list"));
    }

    expertNode.inTensorIds = {GetTensorIdx(tensorMap,
        (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute) ? \
        "intermediate_recv_hiddenstatus" : "in_hiddenstatus"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_weight_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_bias_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_descale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_offset_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_scale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_compress_idx_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_weight_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_bias_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_descale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_offset_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_scale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_compress_idx_expert"),
        GetTensorIdx(tensorMap, "in_expert_array"),
        GetTensorIdx(tensorMap, (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute) ? \
        "intermediate_recv_selected_experts" : "in_selected_experts"),
        GetTensorIdx(tensorMap, (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute) ? \
        "in_expert_array" : "in_expert_weight")};
    if (param.hasMoeEp) {
        expertNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_zero_hot"));
        expertNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_start_expert_idx"));
        expertNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_device_expert_count"));
        expertNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_padding_idx"));
    }
    ATB_SPEED_LOG_DEBUG("Expert Group calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateShuffleIdxDE2(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &gatherNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));

    gatherNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx"),
                                GetTensorIdx(tensorMap, "intermediate_buffer_idx")};
    gatherNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx_2")};

    ATB_SPEED_LOG_DEBUG("CreateShuffleIdxDE");
    return atb::NO_ERROR;
}

atb::Status CreateShuffleWeight(std::map<std::string, uint32_t> &tensorMap,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &node = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(elewiseParam, &node.operation));

    node.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_experts_weight"),
        GetTensorIdx(tensorMap, "intermediate_shuffle_weight")};
    node.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_weight")};

    node.inTensorReshapeFuncs.resize(node.inTensorIds.size());
    node.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    ATB_SPEED_LOG_DEBUG("CreateShuffleWeight");
    return atb::NO_ERROR;
}

atb::Status CreateAllToAllCollect(std::map<std::string, uint32_t> &tensorMap, const DynamicEpMoEParam &param,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    auto &allToAllCollectNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::AllToAllCollectParam allToAllCollectParam;

    allToAllCollectParam.topk = param.topk;
    allToAllCollectParam.numOfExperts = param.numOfExperts;

    allToAllCollectParam.backend = param.backend;
    allToAllCollectParam.hcclComm = param.hcclComm;
    allToAllCollectParam.hasMoeEp = param.hasMoeEp;
    allToAllCollectParam.moeEpRank = param.moeEpRank;
    allToAllCollectParam.moeEpSize = param.moeEpSize;
    allToAllCollectParam.moeEpRankTableFile = param.moeEpRankTableFile;
    allToAllCollectParam.moeEpDomain = param.moeEpDomain;

    allToAllCollectParam.hasMlpTp = param.hasMlpTp;
    allToAllCollectParam.mlpTpRank = param.mlpTpRank;
    allToAllCollectParam.mlpTpSize = param.mlpTpSize;
    allToAllCollectParam.mlpTpRankTableFile = param.mlpTpRankTableFile;
    allToAllCollectParam.mlpTpDomain = param.mlpTpDomain;
    atb_speed::common::CreateAllToAllCollectOperation(allToAllCollectParam, &allToAllCollectNode.operation);
    allToAllCollectNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstatus"),
                                       GetTensorIdx(tensorMap, "intermediate_moe_output"),
                                       GetTensorIdx(tensorMap, "intermediate_shuffle_weight"),
                                       GetTensorIdx(tensorMap, "intermediate_shuffle_idx_1"),
                                       GetTensorIdx(tensorMap, "intermediate_valid_idx")};
    allToAllCollectNode.outTensorIds = {GetTensorIdx(tensorMap, "out_hiddenstates")};
    allToAllCollectNode.inTensorChunks.resize(allToAllCollectNode.inTensorIds.size());
    ATB_SPEED_LOG_DEBUG("allToAllCollectNode calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateDynamicEpMoEOperation(const DynamicEpMoEParam &param, atb::Operation **operation)
{
    ATB_SPEED_LOG_DEBUG("CreateDynamicEpMoEOperation Start");
    atb::GraphParam opGraph;
    opGraph.name = "DynamicEpMoE";
    std::map<std::string, uint32_t> tensorMap = ConstructDynamicEpTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);

    uint64_t nodeCount = 1;
    if (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute && !param.enableLcocAll2All) {
        nodeCount = 9; // 9: ep level2 translated9translated
    }
    opGraph.nodes.resize(nodeCount);

    size_t nodeId = 0;
    if (param.enableLcocAll2All && param.isDynamicEp) {
        // alltoall GMM operation
        /*
            1. InitRoutingQuant
            2. Allgather Matrix
            3. AlltoallGMM
            4  DequantSwigluQuant
            5  GMMAlltoall
            6  MoeTokenUnpermute
        */
        CHECK_OPERATION_STATUS_RETURN(CreateFusedAllToAllMlp(tensorMap, param, nodeId, opGraph));
    } else if (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute) {
        CHECK_OPERATION_STATUS_RETURN(CreateDataPreparation(tensorMap, param, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateAllToAllMeta(tensorMap, param, nodeId, opGraph));

        CHECK_OPERATION_STATUS_RETURN(CreateShuffleIdxDE(tensorMap, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateExpertShuffleIdx(tensorMap, nodeId, opGraph));

        CHECK_OPERATION_STATUS_RETURN(CreateAllToAllDispatch(tensorMap, param, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateMoeMlp(tensorMap, param, nodeId, opGraph));

        CHECK_OPERATION_STATUS_RETURN(CreateShuffleWeight(tensorMap, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateShuffleIdxDE2(tensorMap, nodeId, opGraph));  // translated
        CHECK_OPERATION_STATUS_RETURN(CreateAllToAllCollect(tensorMap, param, nodeId, opGraph));
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateMoeMlp(tensorMap, param, nodeId, opGraph));
    }

    opGraph.inferShapeFunc = [=] (const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (param.enableExpertCumSumOutput) {
            outTensorDescs.at(1) = atb::TensorDesc{};
            outTensorDescs.at(1).format = ACL_FORMAT_ND;
            outTensorDescs.at(1).shape.dimNum = 1;
            outTensorDescs.at(1).dtype = ACL_INT64;
            outTensorDescs.at(1).shape.dims[0] = param.numOfDeviceExperts;
        }
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    ATB_SPEED_LOG_DEBUG("CreateDynamicEpMoEOperation seccess");
    return atb::NO_ERROR;
}
}
} // namespace atb_speed
