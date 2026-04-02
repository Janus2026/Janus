/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include "fused_alltoall_gmm.h"
#include <atb/atb_infer.h>
#include <memory>
#include "operations/aclnn/ops/grouped_matmul_operation.h"
#include "operations/fusion/utils.h"
#include "operations/aclnn/ops/moetoken_unpermute_operation.h"
#include "operations/aclnn/ops/moe_init_routing_quant_operation.h"
#include "operations/aclnn/ops/dequant_swiglu_quant_operation.h"
#include "operations/aclnn/ops/dynamic_quant_operation.h"

namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetAll2AllMatmulInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> all2AllMatmulInTensorCandidates = {
        {"default", {
            "in_hiddenstatus", "in_mlp_gateup_weight_expert", "in_mlp_gateup_descale_expert",
            "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert", "in_mlp_gateup_compress_idx_expert",
            "in_mlp_down_weight_expert", "in_mlp_down_descale_expert", "in_mlp_down_offset_expert",
            "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert", "in_expert_array", "in_selected_experts",
            "in_expert_weight", "in_moe_idx"
            }
        },
    };
    return all2AllMatmulInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetAll2AllMatmulInterTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> all2AllMatmulInterTensorCandidates = {
        {"default", {
            "intermediate_hiddenstates",
            "intermediate_idx",
            "intermediate_group_list",
            "intermediate_dynamic_scale",
            "intermediate_group_list_full",
            "intermediate_gate_up_out",
            "intermediate_quant_swish_out",
            "intermediate_swish_out_scale",
            "intermediate_mlp_out",
            "intermediate_tokens_before_capacity"
            }
        },
    };
    return all2AllMatmulInterTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetAll2AllMatmulOutTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> all2AllMatmulOutTensorCandidates = {
        {"default", {
            "out_hiddenstates"}
        },
    };
    return all2AllMatmulOutTensorCandidates;
}

std::map<std::string, uint32_t> ConstructDynamicEpTensorMap(
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &interTensorNum)
{
    auto all2AllMatmulInTensorCandidates = GetAll2AllMatmulInTensorCandidates();
    auto all2AllMatmulInterTensorCandidates = GetAll2AllMatmulInterTensorCandidates();
    auto all2AllMatmulOutTensorCandidates = GetAll2AllMatmulOutTensorCandidates();
    std::vector<std::string> inTensorList = {};
    std::vector<std::string> interTensorList = {};
    std::vector<std::string> outTensorList = {};
    AddTensorToList(all2AllMatmulInTensorCandidates, "default", inTensorList);
    AddTensorToList(all2AllMatmulInterTensorCandidates, "default", interTensorList);
    AddTensorToList(all2AllMatmulOutTensorCandidates, "default", outTensorList);
    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    interTensorNum = interTensorList.size();
    return GetTensorMap(inTensorList, outTensorList, interTensorList);
}


atb::Status CreateInitRoutingQuant(
    std::map<std::string, uint32_t> &tensorMap, const All2AllMatmulParam &param,
    atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &initRoutingNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::MoeInitRoutingQuantParam initRoutingParam;
    initRoutingParam.topkNum = param.topk;
    initRoutingParam.scaledTopk =
        (param.scaledTopk == atb_speed::common::DEFAULT_TOPK_SCALE)
            ? param.topk : param.scaledTopk;
    initRoutingParam.expertNum = param.numOfExperts;
    initRoutingParam.expertTokensCoutOrCumsumFlag = NUM2;  // translated,translatedexperttranslated
    initRoutingNode.operation = new atb_speed::common::MoeInitRoutingQuantOperation("MoeInitRoutingQuantOperation",
        initRoutingParam);
    initRoutingNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_hiddenstatus"),
        GetTensorIdx(tensorMap, "in_selected_experts")
    };
    initRoutingNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_hiddenstates"),
        GetTensorIdx(tensorMap, "intermediate_idx"),
        GetTensorIdx(tensorMap, "intermediate_group_list"),
        GetTensorIdx(tensorMap, "intermediate_tokens_before_capacity"),
        GetTensorIdx(tensorMap, "intermediate_dynamic_scale")
    };
    ATB_SPEED_LOG_DEBUG("FusedAlltoallGMM Create InitRouting success");
    return atb::NO_ERROR;
}

atb::Status CreateGrouplistAllGather(
    std::map<std::string, uint32_t> &tensorMap, const All2AllMatmulParam &param,
    atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &allGatherGrouplistNode = opGraph.nodes.at(nodeId++);
    atb::infer::AllGatherParam allGatherParam;
    allGatherParam.rank = param.moeEpRank;
    allGatherParam.rankSize = param.moeEpSize;
    allGatherParam.backend = "lccl";
    allGatherParam.rankTableFile = "";
    allGatherParam.commDomain = param.lcclMoeEpDomain;
    allGatherGrouplistNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_list")};
    allGatherGrouplistNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_list_full")};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherGrouplistNode.operation));
    ATB_SPEED_LOG_DEBUG("FusedAlltoallGMM Create AllGather success");
    return atb::NO_ERROR;
}


atb::Status CreateAll2AllMatmul(std::map<std::string, uint32_t> &tensorMap, const All2AllMatmulParam &param,
    atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &allToAllMatmulNode = opGraph.nodes.at(nodeId++);
    atb::infer::LinearParallelParam allToAllMatmulParam;

    allToAllMatmulParam.rank = param.moeEpRank;
    allToAllMatmulParam.rankSize = param.moeEpSize;
    allToAllMatmulParam.backend = "lcoc";
    allToAllMatmulParam.rankTableFile = "";
    allToAllMatmulParam.commDomain = "";
    allToAllMatmulParam.transWeight = param.gateUpTransposeB;
    allToAllMatmulParam.type = atb::infer::LinearParallelParam::ParallelType::ALLTOALLVC_ALL_GATHER_GMM;
    allToAllMatmulParam.quantType = atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_TOKEN;
    allToAllMatmulParam.moeInfo.epSize = param.moeEpSize;
    allToAllMatmulParam.moeInfo.tpSize = 1;
    allToAllMatmulParam.moeInfo.localExpertNums = param.numOfDeviceExperts;
    allToAllMatmulParam.outDataType = aclDataType::ACL_FLOAT16;

    CHECK_OPERATION_STATUS_RETURN(CreateOperation(allToAllMatmulParam, &allToAllMatmulNode.operation));

    allToAllMatmulNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_hiddenstates"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_weight_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_scale_expert"),  // per channel
        GetTensorIdx(tensorMap, "intermediate_dynamic_scale"),  // per token
        GetTensorIdx(tensorMap, "intermediate_group_list_full"),  // expert_per_token_matrix [ep, 256]
        GetTensorIdx(tensorMap, "in_moe_idx")
    };
    allToAllMatmulNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_gate_up_out")};
    ATB_SPEED_LOG_DEBUG("FusedAlltoallGMM Create AllGather success");
    return atb::NO_ERROR;
}


atb::Status CreateSwigluQuant(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &swigluQuantNode = opGraph.nodes.at(nodeId++);
    AclNNDequantSwigluQuantParam aclnnParam;
    aclnnParam.activateLeft = true;
    aclnnParam.quantMode = "dynamic";
    aclnnParam.inTensorsNum = 1;
    swigluQuantNode.operation = new atb_speed::common::DequantSwigluQuantOperation("swigluQuantNode", aclnnParam);
    swigluQuantNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_gate_up_out")};
    swigluQuantNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_quant_swish_out"),
        GetTensorIdx(tensorMap, "intermediate_swish_out_scale")
    };
    ATB_SPEED_LOG_DEBUG("FusedAlltoallGMM Create SwigluQuant success");
    return atb::NO_ERROR;
}

atb::Status CreateMatmulAll2All(std::map<std::string, uint32_t> &tensorMap,
    const All2AllMatmulParam &param, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &matmulAllToAllNode = opGraph.nodes.at(nodeId++);
    atb::infer::LinearParallelParam matmulAllToAllParam;

    matmulAllToAllParam.rank = param.moeEpRank;
    matmulAllToAllParam.rankSize = param.moeEpSize;
    matmulAllToAllParam.backend = "lcoc";
    matmulAllToAllParam.rankTableFile = "";
    matmulAllToAllParam.commDomain = "";
    matmulAllToAllParam.transWeight = param.downTransposeB;
    matmulAllToAllParam.type = atb::infer::LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC;
    matmulAllToAllParam.quantType = atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_TOKEN;
    matmulAllToAllParam.moeInfo.localExpertNums = param.numOfDeviceExperts;
    matmulAllToAllParam.moeInfo.tpSize = 1;
    matmulAllToAllParam.moeInfo.epSize = param.moeEpSize;
    matmulAllToAllParam.outDataType = aclDataType::ACL_FLOAT16;

    CHECK_OPERATION_STATUS_RETURN(CreateOperation(matmulAllToAllParam, &matmulAllToAllNode.operation));

    matmulAllToAllNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_quant_swish_out"),
        GetTensorIdx(tensorMap, "in_mlp_down_weight_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_scale_expert"),
        GetTensorIdx(tensorMap, "intermediate_swish_out_scale"),
        GetTensorIdx(tensorMap, "intermediate_group_list_full"),
        GetTensorIdx(tensorMap, "intermediate_idx")
        };
    matmulAllToAllNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out")};

    return atb::NO_ERROR;
}

atb::Status CreateMoeTokenUnpermute(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph,  size_t &nodeId)
{
    atb::Node &unpermuteNode = opGraph.nodes.at(nodeId++);
    unpermuteNode.operation = new atb_speed::common::MoeTokenUnpermuteOperation("MoeTokenUnpermuteNode");
    unpermuteNode.outTensorIds = {GetTensorIdx(tensorMap, "out_hiddenstates")};
    unpermuteNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out"),
                                 GetTensorIdx(tensorMap, "intermediate_idx"),
                                 GetTensorIdx(tensorMap, "in_expert_weight")};

    ATB_SPEED_LOG_DEBUG("UnpermuteNode calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateAll2AllMatmulOperation(const All2AllMatmulParam &param, atb::Operation **operation)
{
    ATB_SPEED_LOG_DEBUG("CreateAll2AllMatmulOperation Start");
    atb::GraphParam opGraph;
    opGraph.name = "All2AllMatmul";
    std::map<std::string, uint32_t> tensorMap = ConstructDynamicEpTensorMap(
        opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);

    uint64_t nodeCount = 6;
    opGraph.nodes.resize(nodeCount);

    size_t nodeId = 0;
    CHECK_OPERATION_STATUS_RETURN(CreateInitRoutingQuant(tensorMap, param, opGraph, nodeId));
    CHECK_OPERATION_STATUS_RETURN(CreateGrouplistAllGather(tensorMap, param, opGraph, nodeId));
    CHECK_OPERATION_STATUS_RETURN(CreateAll2AllMatmul(tensorMap, param, opGraph, nodeId));
    CHECK_OPERATION_STATUS_RETURN(CreateSwigluQuant(tensorMap, opGraph, nodeId));
    CHECK_OPERATION_STATUS_RETURN(CreateMatmulAll2All(tensorMap, param, opGraph, nodeId));
    CHECK_OPERATION_STATUS_RETURN(CreateMoeTokenUnpermute(tensorMap, opGraph, nodeId));
    
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    ATB_SPEED_LOG_DEBUG("CreateAll2AllMatmulOperation seccess");
    return atb::NO_ERROR;
}
}
} // namespace atb_speed