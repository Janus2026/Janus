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
#include "moe_mlp.h"
#include <atb/atb_infer.h>
#include <iostream>
#include <memory>
#include <algorithm>
#include "operations/fusion/moe/integrated_gmm.h"
#include "operations/fusion/moe/ep/expert_filter.h"
#include "operations/aclnn/ops/finalize_routing_operation.h"
#include "operations/aclnn/ops/moe_init_routing_operation.h"
#include "operations/aclnn/ops/moe_init_routing_quant_operation.h"
#include "operations/aclnn/ops/moe_init_routing_operation_v3.h"
#include "operations/aclnn/ops/moe_compute_expert_tokens_operation.h"
#include "operations/aclnn/ops/moetoken_unpermute_operation.h"
#include "operations/aclnn/ops/inplacemasked_filltensor_operation.h"
#include "operations/aclnn/ops/grouped_matmul_operation.h"
#include "atb_speed/base/event_manager.h"
#include "operations/aclnn/utils/utils.h"
#include "operations/aclnn/ops/moe_distribute_combine_operation.h"
#include "operations/aclnn/ops/moe_distribute_dispatch_operation.h"
#include "operations/aclnn/ops/quant_gmm_dequant_operation.h"
#include "operations/aclnn/ops/moe_distribute_combine_v2_operation.h"
#include "operations/aclnn/ops/moe_distribute_dispatch_v2_operation.h"
#include "operations/aclnn/ops/len_operation.h"
#include "operations/aclnn/ops/minimum_operation.h"
#include "operations/aclnn/ops/swi_glu_operation.h"
#include "operations/aclnn/ops/moe_grouped_matmul_weight_nz_operation.h"

namespace atb_speed {
namespace common {


int CalcUpGmmQuantType(const MoeMlpParam &param)
{
    int gmmQuantType = 0;
    int tempQuantType = atb_speed::common::GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED ?
            param.packQuantType : param.denseQuantType,
        param.moeLinearQuantType[IntegratedGmmIdx::MOE_MLP_GATE_IDX], false);
    switch (tempQuantType) {
        case LinearQuantType::NO_QUANT:
            gmmQuantType = GmmQuantType::NONE;
            break;
        case LinearQuantType::LINEAR_W8A8_DYNAMIC_QUANT:
        case LinearQuantType::LINEAR_W8A8_DYNAMIC_DEQUANT:
            gmmQuantType = GmmQuantType::W8A8_TOKEN;
            break;
        case LinearQuantType::W8A16:
            gmmQuantType = GmmQuantType::W8A16_CHANNEL;
            break;
        case LinearQuantType::W4A16:
            gmmQuantType = GmmQuantType::W4A16_CHANNEL;
            break;
        case LinearQuantType::LINEAR_W4A8_DYNAMIC_DEQUANT:
        case LinearQuantType::LINEAR_W4A8_DYNAMIC_QUANT:
            gmmQuantType = GmmQuantType::W4A8_GROUP;
            break;
        default:
            gmmQuantType = GmmQuantType::W8A8_CHANNEL;
            break;
    }
    return gmmQuantType;
}

bool IsGMMSwigluQuant(const int gmmQuantType, const MoeMlpParam &param)
{
    return gmmQuantType == GmmQuantType::W8A8_TOKEN && param.enableGMMSwigluQuant;
}

std::map<std::string, std::vector<std::string>> GetMoeMlpInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> moeMlpInTensorCandidates = {
        {"default", {
            "in_hiddenstates", "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert",
            "in_mlp_gateup_descale_expert", "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert",
            "in_mlp_gateup_compress_idx_expert", "in_mlp_down_weight_expert",
            "in_mlp_down_bias_expert", "in_mlp_down_descale_expert", "in_mlp_down_offset_expert",
            "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert", "in_expert_array", "in_selected_experts",
            "in_expert_weight"},
        },
        {"ep", {
            "in_zero_hot", "in_start_expert_idx", "in_device_expert_count", "in_padding_idx"}
        }
    };
    return moeMlpInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetMoeMlpInterTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> moeMlpInterTensorCandidates = {
        {"default", {
            "intermediate_idx", "intermediate_weight_idx", "intermediate_dummy_zero",
            "intermediate_dummy_one", "intermediate_rev_idx", "intermediate_group_list",
            "intermediate_sorted_hiddenstates", "intermediate_rev_sorted_hiddenstates",
            "intermediate_matmul_gate_up_out", "intermediate_swish_out", "intermediate_mlp_out",
            "intermediate_mlp_out_weighted", "intermediate_sorted_weight"}
        },
        {"default_w8a8_token", {
            "intermediate_idx", "intermediate_weight_idx", "intermediate_dummy_zero",
            "intermediate_dummy_one", "intermediate_rev_idx", "intermediate_group_list",
            "intermediate_sorted_hiddenstates", "intermediate_rev_sorted_hiddenstates",
            "intermediate_mlp_out", "intermediate_mlp_out_weighted", "intermediate_sorted_weight"}
        },
        {"enableFusedRouting", {
            "intermediate_idx", "intermediate_sorted_hiddenstates",
            "intermediate_matmul_gate_up_out", "intermediate_swish_out", "intermediate_mlp_out"}
        },
        {"enable_mc2", {
            "intermediate_ep_recv_counts", "intermediate_tp_recv_counts", "intermediate_gmm0_deqscale",
            "intermediate_expand_expert_weight"}
        },
        {"enableFusedRouting_w8a8_token", {
            "intermediate_idx",  "intermediate_sorted_hiddenstates", "intermediate_mlp_out"}
        },
        {"disable_swiglu", {
            "intermediate_matmul_gate_out", "intermediate_matmul_up_out", "intermediate_swish_out_internal"}
        },
        {"enable_init_quant", {
            "intermediate_tokens_before_capacity", "intermediate_sorted_hiddenstates_dequant_scale"}
        },
        {"enable_init_routing_v3", {
            "intermediate_sorted_hiddenstates_dequant_scale"}
        },
        {"enable_swiglu_quant", {
            "intermedaite_swiglu_dequant_scale"}
        },
        {"ep", {
            "intermediate_group_list_sliced", "intermediate_expert_weight"}
        },
        {"dynanmic_ep", {
            "intermediate_selected_experts"}
        },
        {"gating_dp", {
            "intermediate_group_list_sliced", "intermediate_expert_weight", "intermediate_selected_experts"}
        },
        {"initrouting_cutoff", {
            "intermediate_sorted_hiddenstates_len", "intermediate_group_list_filtered"}}
    };
    return moeMlpInterTensorCandidates;
}

atb::Status AddIntermediateTensor(const MoeMlpParam &param, std::vector<std::string>& interTensorList)
{
    auto moeMlpInterTensorCandidates = GetMoeMlpInterTensorCandidates();
    bool isGMMSwigluQuant = IsGMMSwigluQuant(CalcUpGmmQuantType(param), param);

    if (param.enableFusedRouting) {
        // All init_routing variants' logic is in this branch
        if (isGMMSwigluQuant) {
            // Branch 1: gmm_swiglu_quant fusion
            AddTensorToList(moeMlpInterTensorCandidates, "enableFusedRouting_w8a8_token", interTensorList);
        } else {
            // Branch 2: no gmm_swiglu
            AddTensorToList(moeMlpInterTensorCandidates, "enableFusedRouting", interTensorList);
        }

        if (param.enableInitRoutingV3) {
            // Branch 1(init routing v3): quant/no quant + init_routing_v3
            AddTensorToList(moeMlpInterTensorCandidates, "enable_init_routing_v3", interTensorList);
        } else if (param.enableInitQuant && !param.enableInitRoutingV3){
            // Branch 2(default quant): quant + init_routing_quant
            AddTensorToList(moeMlpInterTensorCandidates, "enable_init_quant", interTensorList);
        }
        
        if (param.enableMoeDistribute) {
            // Using mc2 when enable moe distribute.
            AddTensorToList(moeMlpInterTensorCandidates, "enable_mc2", interTensorList);
            interTensorList.push_back("intermediate_group_list");
        } else {
            // Otherwise, intermediate_group_list_int64 is required. 
            // 'Cause init_routing_v3 already gives int 64 output, no need to add again.
            interTensorList.push_back("intermediate_group_list_int64");
            if (param.enableInitRoutingV3){
                // already output int64
            } else {
                interTensorList.push_back("intermediate_group_list");
            }
        }
    } else {
        AddTensorToList(moeMlpInterTensorCandidates, isGMMSwigluQuant ?
            "default_w8a8_token" : "default", interTensorList);
    }

    if (param.enableSwigluQuant) {
        AddTensorToList(moeMlpInterTensorCandidates, "enable_swiglu_quant", interTensorList);
    }
    if (!param.supportSwiGLU) {
        AddTensorToList(moeMlpInterTensorCandidates, "disable_swiglu", interTensorList);
    }
    if (param.hasMoeEp && !param.enableMoeDistribute) {
        if (!param.enableGatingDp) {
            AddTensorToList(moeMlpInterTensorCandidates, "ep", interTensorList);
            if (!param.shiftedTopK) {
                AddTensorToList(moeMlpInterTensorCandidates, "dynanmic_ep", interTensorList);
            }
        } else {
            AddTensorToList(moeMlpInterTensorCandidates, "gating_dp", interTensorList);
        }
    }
    if (param.enableInitRoutingCutoff) {
        AddTensorToList(moeMlpInterTensorCandidates, "initrouting_cutoff", interTensorList);
    }
    return atb::NO_ERROR;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const MoeMlpParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &interTensorNum)
{
    auto moeMlpInTensorCandidates = GetMoeMlpInTensorCandidates();
    std::vector<std::string> inTensorList = {};
    std::vector<std::string> interTensorList = {};
    std::vector<std::string> outTensorList = {"out_moe_mlp_result"};
    AddTensorToList(moeMlpInTensorCandidates, "default", inTensorList);
    if (param.hasMoeEp) {
        AddTensorToList(moeMlpInTensorCandidates, "ep", inTensorList);
    }

    AddIntermediateTensor(param, interTensorList);

    if (param.enableExpertCumSumOutput) {
        if (param.enableFusedRouting && !param.enableMoeDistribute) {
            interTensorList.erase(std::remove(interTensorList.begin(), interTensorList.end(),
                "intermediate_group_list_int64"), interTensorList.end());
            outTensorList.push_back("intermediate_group_list_int64");
        } else {
            interTensorList.erase(std::remove(interTensorList.begin(), interTensorList.end(),
                "intermediate_group_list"), interTensorList.end());
            outTensorList.push_back("intermediate_group_list");
        }
    }
    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    interTensorNum = interTensorList.size();
    return GetTensorMap(inTensorList, outTensorList, interTensorList);
}

atb::Status CreateExpertFilter(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph)
{
    atb::Node expertFilterNode;
    atb_speed::common::ExpertFilterParam expertFilterParam;
    expertFilterParam.numOfExperts = param.numOfExperts;
    expertFilterParam.deviceExpert = param.deviceExpert;
    expertFilterParam.shiftedTopK = param.shiftedTopK;
    expertFilterParam.isBF16 = param.isBF16;
    expertFilterParam.enableGatingDp = param.enableGatingDp;
    atb_speed::common::CreateExpertFilterOperation(expertFilterParam, &expertFilterNode.operation);

    expertFilterNode.inTensorIds = {GetTensorIdx(tensorMap, "in_selected_experts"),
                                    GetTensorIdx(tensorMap, "in_expert_weight"),
                                    GetTensorIdx(tensorMap, "in_start_expert_idx"),
                                    GetTensorIdx(tensorMap, "in_device_expert_count"),
                                    GetTensorIdx(tensorMap, "in_zero_hot")};

    if (param.shiftedTopK && !param.enableGatingDp) {
        expertFilterNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_expert_weight")};
    } else {
        expertFilterNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_selected_experts"),
                                         GetTensorIdx(tensorMap, "intermediate_expert_weight")};
    }
    opGraph.nodes.push_back(expertFilterNode);
    ATB_SPEED_LOG_DEBUG("InitRouting calculation success");
    return atb::NO_ERROR;
}

// Step 1: hidden state permutation
atb::Status CreateInitRoutingQuant(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph)
{
    atb::Node initRoutingNode;
    if (param.enableInitRoutingV3) {
        atb_speed::common::MoeInitRoutingV3Param initRoutingParam;
        initRoutingParam.topkNum = param.topk;
        initRoutingParam.scaledTopk = param.scaledTopk;
        initRoutingParam.enableInitRoutingCutoff = param.enableInitRoutingCutoff;
        initRoutingParam.expertNum = param.numOfExperts;
        initRoutingParam.quantMode = 1;
        initRoutingParam.expertTokensNumType = 2;
        int gmmQuantType = CalcUpGmmQuantType(param);
        initRoutingNode.operation = new atb_speed::common::MoeInitRoutingV3Operation("MoeInitRoutingQuantOperation" \
                                                                                + std::to_string(gmmQuantType),
                                                                               initRoutingParam);
    } else {
        atb_speed::common::MoeInitRoutingQuantParam initRoutingParam;
        initRoutingParam.topkNum = param.topk;
        /// deepseektranslatedscaledTopktranslated,translated
        initRoutingParam.scaledTopk = param.scaledTopk;
        initRoutingParam.enableInitRoutingCutoff = param.enableInitRoutingCutoff;
        initRoutingParam.expertNum = param.numOfExperts;
        int gmmQuantType = CalcUpGmmQuantType(param);
        if (gmmQuantType == GmmQuantType::W4A8_GROUP) {
            initRoutingParam.expertTokensCoutOrCumsumFlag = 2; // 2 : W4A8_GROUP_Mutmal translated
            initRoutingParam.enableInitRoutingCutoff = false;
        }
        initRoutingNode.operation = new atb_speed::common::MoeInitRoutingQuantOperation("MoeInitRoutingQuantOperation" \
                                                                                + std::to_string(gmmQuantType),
                                                                               initRoutingParam);
    }
    initRoutingNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"),
                                   GetTensorIdx(tensorMap, param.hasMoeEp && !param.shiftedTopK ? \
                                   "intermediate_selected_experts" : (param.enableGatingDp ? \
                                   "intermediate_selected_experts" : "in_selected_experts"))};
    if (param.enableInitRoutingV3) {
        initRoutingNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates"),
                               GetTensorIdx(tensorMap, "intermediate_idx"),
                               GetTensorIdx(tensorMap, (!param.enableMoeDistribute)?"intermediate_group_list_int64":"intermediate_group_list"),
			       // GetTensorIdx(tensorMap, "intermediate_tokens_before_capacity"),
                               GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates_dequant_scale")};
    } else {
        initRoutingNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates"),
                               GetTensorIdx(tensorMap, "intermediate_idx"),
                               GetTensorIdx(tensorMap, "intermediate_group_list"),
                               GetTensorIdx(tensorMap, "intermediate_tokens_before_capacity"),
                               GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates_dequant_scale")};
    }
    opGraph.nodes.push_back(initRoutingNode);
    ATB_SPEED_LOG_DEBUG("InitRouting calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateInitRouting(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph)
{
    atb::Node initRoutingNode;
    atb_speed::common::MoeInitRoutingParam initRoutingParam;
    initRoutingParam.topkNum = param.topk;
    /// deepseektranslatedscaledTopktranslated,translated
    initRoutingParam.scaledTopk = param.scaledTopk;
    initRoutingParam.enableInitRoutingCutoff = param.enableInitRoutingCutoff;
    initRoutingParam.expertNum = param.numOfExperts;
    int gmmQuantType = CalcUpGmmQuantType(param);
    if (gmmQuantType == GmmQuantType::W4A8_GROUP) {
        initRoutingParam.expertTokensCoutOrCumsumFlag = 2; // 2 : W4A8_GROUP_Mutmal translated
        initRoutingParam.enableInitRoutingCutoff = false;
    }
    initRoutingNode.operation = new atb_speed::common::MoeInitRoutingOperation("MoeInitRoutingOperation" \
                                                                                + std::to_string(gmmQuantType),
                                                                               initRoutingParam);
    initRoutingNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"),
                                   GetTensorIdx(tensorMap, param.hasMoeEp && !param.shiftedTopK ? \
                                   "intermediate_selected_experts" : (param.enableGatingDp ? \
                                   "intermediate_selected_experts" : "in_selected_experts"))};
    initRoutingNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates"),
                               GetTensorIdx(tensorMap, "intermediate_idx"),
                               GetTensorIdx(tensorMap, "intermediate_group_list")};
    opGraph.nodes.push_back(initRoutingNode);
    ATB_SPEED_LOG_DEBUG("InitRouting calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateInitRoutingV3NoQuant(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph)
{
    atb::Node initRoutingNode;
    atb_speed::common::MoeInitRoutingV3Param initRoutingParam;
    initRoutingParam.topkNum = param.topk;
    /// deepseektranslatedscaledTopktranslated,translated
    initRoutingParam.scaledTopk = param.scaledTopk;
    initRoutingParam.enableInitRoutingCutoff = param.enableInitRoutingCutoff;
    initRoutingParam.expertNum = param.numOfExperts;
    initRoutingParam.expertTokensNumType = 2;
    initRoutingParam.quantMode = -1;

    initRoutingNode.operation = new atb_speed::common::MoeInitRoutingV3Operation("MoeInitRoutingOperation", initRoutingParam);
    initRoutingNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"),
                                   GetTensorIdx(tensorMap, param.hasMoeEp && !param.shiftedTopK ? \
                                   "intermediate_selected_experts" : (param.enableGatingDp ? \
                                   "intermediate_selected_experts" : "in_selected_experts"))};
    initRoutingNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates"),
                               GetTensorIdx(tensorMap, "intermediate_idx"),
                               GetTensorIdx(tensorMap, (!param.enableMoeDistribute)?"intermediate_group_list_int64":"intermediate_group_list"),
    			       GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates_dequant_scale")}; // not used
    opGraph.nodes.push_back(initRoutingNode);
    ATB_SPEED_LOG_DEBUG("InitRouting calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateComputeExpertSlice(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph)
{
    atb::Node sliceNode;
    atb::infer::SliceParam sliceParam;
    sliceParam.offsets.resize(1);
    sliceParam.offsets[0] = 0;
    sliceParam.size.resize(1);
    sliceParam.size[0] = param.numOfDeviceExperts;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(sliceParam, &sliceNode.operation));
    sliceNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_list")};
    sliceNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_list_sliced")};
    opGraph.nodes.push_back(sliceNode);
    ATB_SPEED_LOG_DEBUG("sliceNode calculation success");
    return atb::NO_ERROR;
}


atb::Status CreateExpandedXLen(
    std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph)
{
    atb::Node lenNode;
    lenNode.operation = new atb_speed::common::LenOperation("LenOperation");
    lenNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates")};
    lenNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates_len")};
    opGraph.nodes.push_back(lenNode);
    ATB_SPEED_LOG_DEBUG("LenOperation calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateGroupListFilter(
    std::map<std::string, uint32_t> &tensorMap,
    const MoeMlpParam &param,
    atb::GraphParam &opGraph)
{
    atb::Node minimumNode;
    minimumNode.operation = new atb_speed::common::MinimumOperation("MinimumOperation");
    minimumNode.inTensorIds = {
        GetTensorIdx(tensorMap, param.hasMoeEp ? "intermediate_group_list_sliced" : "intermediate_group_list"),
        GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates_len")};
    minimumNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_list_filtered")};
    opGraph.nodes.push_back(minimumNode);
    ATB_SPEED_LOG_DEBUG("MinimumOperation calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateCast(std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph)
{
    atb::Node castNode;
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_INT64;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {
        GetTensorIdx(tensorMap, param.enableInitRoutingCutoff ? "intermediate_group_list_filtered" : (
            param.hasMoeEp ? "intermediate_group_list_sliced" : "intermediate_group_list")
        )
    };
    castNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_list_int64")};
    opGraph.nodes.push_back(castNode);
    ATB_SPEED_LOG_DEBUG("Cast calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateGating(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param, std::shared_ptr<int64_t> batchDimPtr,
    atb::GraphParam &opGraph)
{
    atb::Node gatingNode;
    CHECK_PARAM_NE(param.topk, 0);
    CHECK_PARAM_NE(param.numOfExperts, 0);
    atb::infer::GatingParam gatingParam;
    gatingParam.topkExpertNum = param.topk;
    gatingParam.cumSumNum = param.numOfExperts;
    gatingParam.cumSumInt64 = true;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatingParam, &gatingNode.operation));
    gatingNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_selected_experts"), GetTensorIdx(tensorMap, "in_expert_array")};
    gatingNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_idx"), GetTensorIdx(tensorMap, "intermediate_group_list"),
        GetTensorIdx(tensorMap, "intermediate_weight_idx")};
    gatingNode.inTensorReshapeFuncs.resize(gatingNode.inTensorIds.size());
    gatingNode.inTensorReshapeFuncs[0] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    opGraph.nodes.push_back(gatingNode);
    ATB_SPEED_LOG_DEBUG("Gating calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateGather0(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node gatherNode0;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode0.operation));
    gatherNode0.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"), GetTensorIdx(tensorMap, "intermediate_idx")};
    gatherNode0.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates")};
    opGraph.nodes.push_back(gatherNode0);
    ATB_SPEED_LOG_DEBUG("Gather0 calculation success");
    return atb::NO_ERROR;
}

// Step 2: grouped matmul calculation & activation
atb::Status CreateGmm(std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph, const MoeMlpParam &param)
{
    atb::Node gmmNode;
    atb_speed::common::IntegratedGmmParam gmmParam;
    gmmParam.hasBias = (param.packQuantType == atb_speed::common::PackQuantType::ALL_W4A8) ? \
        true : param.hasBias;
    gmmParam.isUp = true;
    gmmParam.moeLinearQuantType = param.moeLinearQuantType;
    gmmParam.packQuantType = param.packQuantType;
    gmmParam.transposeB = param.gateUpTransposeB;
    gmmParam.downTransposeB = param.downTransposeB;
    gmmParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    gmmParam.quantGroupSize = param.quantGroupSize;
    gmmParam.enableGMMSwigluQuant = param.enableGMMSwigluQuant;
    gmmParam.enableAtlasGMMFused = param.enableAtlasGMMFused;
    gmmParam.enableIndexGmm = param.enableIndexGmm;
    gmmParam.enableInitRoutingV3 = param.enableInitRoutingV3;
    if (param.enableInitQuant || (param.enableMoeDistribute &&
        (param.packQuantType == atb_speed::common::PackQuantType::ALL_W8A8_DYNAMIC \
            || param.packQuantType == atb_speed::common::PackQuantType::ALL_W4A8))) {
        gmmParam.skipQuant = true;
    }
    if (param.enableCVOverlap) {gmmParam.enableCVOverlap = true;}
    CHECK_OPERATION_STATUS_RETURN(CreateIntegratedGmmOperation(gmmParam, &gmmNode.operation));
    gmmNode.inTensorIds = {};
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_weight_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_bias_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_descale_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_offset_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_scale_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_compress_idx_expert"));
    if (param.enableFusedRouting && !param.enableMoeDistribute) {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list_int64"));
    } else {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list"));
    }
    bool isGMMSwigluQuant = IsGMMSwigluQuant(CalcUpGmmQuantType(param), param);
    if (isGMMSwigluQuant) {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_weight_expert"));
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_bias_expert"));
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_descale_expert"));
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_offset_expert"));
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_scale_expert"));
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_compress_idx_expert"));
    }
    gmmNode.outTensorIds = {GetTensorIdx(tensorMap, isGMMSwigluQuant ? "intermediate_mlp_out" :
        "intermediate_matmul_gate_up_out")};
    if (gmmParam.skipQuant) {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, param.enableMoeDistribute ? "intermediate_gmm0_deqscale"
            : "intermediate_sorted_hiddenstates_dequant_scale"));
    }
    opGraph.nodes.push_back(gmmNode);
    ATB_SPEED_LOG_DEBUG("GMM calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateCustomGmmWeightNZ(std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph, const MoeMlpParam &param)
{
    atb::Node gmmNode;
    gmmNode.operation = new atb_speed::common::MoeGroupedMatmulWeightNZOperation("MoeGmmWeightNZOperation", param.gateUpTransposeB);
    gmmNode.inTensorIds = {};
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_weight_expert"));
    if (param.enableFusedRouting && !param.enableMoeDistribute) {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list_int64"));
    } else {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list"));
    }
    gmmNode.outTensorIds = {GetTensorIdx(tensorMap,  "intermediate_matmul_gate_up_out")};
    opGraph.nodes.push_back(gmmNode);
    ATB_SPEED_LOG_DEBUG("GMM calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateCustomGmm1WeightNZ(std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph, const MoeMlpParam &param)
{
    atb::Node gmmDownNode;
    gmmDownNode.operation = new atb_speed::common::MoeGroupedMatmulWeightNZOperation("MoeGmm1WeightNZOperation", param.downTransposeB);
    gmmDownNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out")};
    gmmDownNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out"),
                               GetTensorIdx(tensorMap, "in_mlp_down_weight_expert")};

    if (param.enableFusedRouting && !param.enableMoeDistribute) {
        gmmDownNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list_int64"));
    } else {
        gmmDownNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list"));
    }

    opGraph.nodes.push_back(gmmDownNode);
    ATB_SPEED_LOG_DEBUG("GMM calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateQuantGMMDequant(
    std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph, const MoeMlpParam &param)
{
    atb::Node quantGmmDequantNode;
    atb_speed::common::AclNNQuantGMMDequantParam aclnnParam;
    aclnnParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    aclnnParam.transposeB = true; // param.gateUpTransposeB;
    aclnnParam.quantMode = "pertoken";
    quantGmmDequantNode.operation = new atb_speed::common::QuantGMMDequantOperation("QuantGMMDequantOperation",
                                                                               aclnnParam);
    quantGmmDequantNode.inTensorIds = { // translated
        GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_weight_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_scale_expert"),
    };
    if (param.enableFusedRouting) {
        quantGmmDequantNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list_int64"));
    } else {
        quantGmmDequantNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list"));
    }
    quantGmmDequantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_up_out")};
    quantGmmDequantNode.inTensorReshapeFuncs.resize(quantGmmDequantNode.inTensorIds.size());
    quantGmmDequantNode.inTensorReshapeFuncs[2] = [param](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // [256, 4096, 1] -> [256, 4096]
        newShape.dims[0] = param.numOfExperts;
        newShape.dims[1] = oldShape.dims[0] / param.numOfExperts * oldShape.dims[1];
    };
    opGraph.nodes.push_back(quantGmmDequantNode);
    ATB_SPEED_LOG_DEBUG("QuantGMMDequant calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateActivation(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node swishNode;
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(activationParam, &swishNode.operation));
    swishNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_up_out")};
    swishNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out")};
    opGraph.nodes.push_back(swishNode);
    ATB_SPEED_LOG_DEBUG("Activation calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateAclnnSwiglu(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node swishNode;
    swishNode.operation = new atb_speed::common::SwiGluOperation("Swiglu", -1);
    swishNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_up_out")};
    swishNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out")};
    opGraph.nodes.push_back(swishNode);
    ATB_SPEED_LOG_DEBUG("Activation calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateActivationQuant(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node swigluQuantNode;
    atb::infer::SwigluQuantParam swigluQuantParam;
    swigluQuantParam.quantType = atb::infer::SwigluQuantParam::QuantType::QUANT_TYPE_PER_TOKEN;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(swigluQuantParam, &swigluQuantNode.operation));
    swigluQuantNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_up_out")};
    swigluQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out"),
                                    GetTensorIdx(tensorMap, "intermedaite_swiglu_dequant_scale")};
    opGraph.nodes.push_back(swigluQuantNode);
    ATB_SPEED_LOG_DEBUG("ActivationQuant calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSplit(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node splitNode;
    atb::infer::SplitParam splitParam = {1, 2};
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(splitParam, &splitNode.operation));
    splitNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_up_out")};
    splitNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_matmul_gate_out"), GetTensorIdx(tensorMap, "intermediate_matmul_up_out")};
    opGraph.nodes.push_back(splitNode);
    ATB_SPEED_LOG_DEBUG("Split calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateActivationO(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node swishNodeO;
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(activationParam, &swishNodeO.operation));
    swishNodeO.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_out")};
    swishNodeO.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out_internal")};
    opGraph.nodes.push_back(swishNodeO);
    ATB_SPEED_LOG_DEBUG("Activation calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateElewiseMul(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node mulNode;
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(elewiseParam, &mulNode.operation));
    mulNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_swish_out_internal"),
        GetTensorIdx(tensorMap, "intermediate_matmul_up_out")};
    mulNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out")};
    opGraph.nodes.push_back(mulNode);
    ATB_SPEED_LOG_DEBUG("ElewiseMul0 calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateGmm1(std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph, const MoeMlpParam &param)
{
    atb::Node gmmDownNode;
    atb_speed::common::IntegratedGmmParam gmmParam;
    gmmParam.hasBias = param.hasBias;
    gmmParam.isUp = false;
    gmmParam.moeLinearQuantType = param.moeLinearQuantType;
    gmmParam.packQuantType = param.packQuantType;
    gmmParam.transposeB = param.downTransposeB;
    gmmParam.enableIndexGmm = param.enableIndexGmm;
    if (param.packQuantType == atb_speed::common::PackQuantType::ALL_W4A8) {
        gmmParam.hasBias = true;
    }
    gmmParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    if (param.enableSwigluQuant) {gmmParam.skipQuant = true;}
    CHECK_OPERATION_STATUS_RETURN(CreateIntegratedGmmOperation(gmmParam, &gmmDownNode.operation));
    gmmDownNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out")};
    gmmDownNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out"),
                               GetTensorIdx(tensorMap, "in_mlp_down_weight_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_bias_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_descale_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_offset_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_scale_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_compress_idx_expert")};
    if (param.enableFusedRouting && !param.enableMoeDistribute) {
        gmmDownNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list_int64"));
    } else {
        gmmDownNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list"));
    }
    if (param.enableSwigluQuant) {
        gmmDownNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermedaite_swiglu_dequant_scale"));
    }
    opGraph.nodes.push_back(gmmDownNode);
    ATB_SPEED_LOG_DEBUG("GMM calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateQuantGMMDequant1(
    std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph, const MoeMlpParam &param)
{
    atb::Node quantGmmDequantDownNode;
    atb_speed::common::AclNNQuantGMMDequantParam aclnnParam;
    aclnnParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    aclnnParam.transposeB = param.downTransposeB;
    aclnnParam.quantMode = "pertoken";
    quantGmmDequantDownNode.operation = new atb_speed::common::QuantGMMDequantOperation("QuantGMMDequantOperation",
                                                                               aclnnParam);
    quantGmmDequantDownNode.inTensorIds = { // translated
        GetTensorIdx(tensorMap, "intermediate_swish_out"),
        GetTensorIdx(tensorMap, "in_mlp_down_weight_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_scale_expert"),
    };
    if (param.enableFusedRouting) {
        quantGmmDequantDownNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list_int64"));
    } else {
        quantGmmDequantDownNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list"));
    }
    quantGmmDequantDownNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out")};
    quantGmmDequantDownNode.inTensorReshapeFuncs.resize(quantGmmDequantDownNode.inTensorIds.size());
    quantGmmDequantDownNode.inTensorReshapeFuncs[2] = [param](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // [256, 4096, 1] -> [256, 4096]
        newShape.dims[0] = param.numOfExperts;
        newShape.dims[1] = oldShape.dims[0] / param.numOfExperts * oldShape.dims[1];
    };
    opGraph.nodes.push_back(quantGmmDequantDownNode);
    ATB_SPEED_LOG_DEBUG("QuantGMMDequant1 calculation success");
    return atb::NO_ERROR;
}

// Step 3: hidden state reduction
atb::Status CreateMoeTokenUnpermute(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param, atb::GraphParam &opGraph)
{
    atb::Node unpermuteNode;
    unpermuteNode.operation = new atb_speed::common::MoeTokenUnpermuteOperation("MoeTokenUnpermuteNode");
    unpermuteNode.outTensorIds = {GetTensorIdx(tensorMap, "out_moe_mlp_result")};
    unpermuteNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out"),
                                 GetTensorIdx(tensorMap, "intermediate_idx"),
                                // shiftedTopKtranslated
                                 GetTensorIdx(tensorMap, param.hasMoeEp && !param.shiftedTopK ? \
                                 "intermediate_expert_weight" : (param.enableGatingDp) ? \
                                 "intermediate_expert_weight" : "in_expert_weight")};
    opGraph.nodes.push_back(unpermuteNode);
    ATB_SPEED_LOG_DEBUG("UnpermuteNode calculation success");
    return atb::NO_ERROR;
}

template<typename DispatchParam>
atb::Status SetMoeDistributeDispatchParam(DispatchParam &dispatchParam, const MoeMlpParam &param)
{
    dispatchParam.epRankId = param.moeEpRank;
    dispatchParam.epRankSize = param.moeEpSize;
    dispatchParam.epCommName = param.moeEpDomain;
    dispatchParam.maxDecodeDpTokenSize = param.maxDecodeDpTokenSize;
    dispatchParam.moeExpertNum = param.numOfExperts;
    dispatchParam.localMoeExpertNum = param.numOfDeviceExperts;
    dispatchParam.sharedExpertRankNum = param.numDanglingSharedExperts;
    dispatchParam.topk = param.topk;
    if (param.packQuantType == atb_speed::common::PackQuantType::ALL_W8A8_DYNAMIC ||
        param.packQuantType == atb_speed::common::PackQuantType::ALL_W8A8_DYNAMIC_ANTI) {
        dispatchParam.quantMode = 2; // 2: translated2
        dispatchParam.isQuant = true;
        dispatchParam.quantSmooth = false;
    } else {
        dispatchParam.quantMode = 0; // 0translated
        dispatchParam.isQuant = false;
        dispatchParam.quantSmooth = false;
    }
    return atb::NO_ERROR;
}

atb::Status CreateMoeDistributeDispatch(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph)
{
    atb::Node moeDistributeDispatchNode;
    if (param.enableDispatchCombineV2) {
        atb_speed::common::MoeDistributeDispatchV2Param dispatchParam;
        SetMoeDistributeDispatchParam(dispatchParam, param);
        moeDistributeDispatchNode.operation = new atb_speed::common::MoeDistributeDispatchV2Operation(
                std::string("MoeDistributeDispatchV2") + std::to_string(param.packQuantType), dispatchParam);
    } else {
        atb_speed::common::MoeDistributeDispatchParam dispatchParam;
        SetMoeDistributeDispatchParam(dispatchParam, param);
        moeDistributeDispatchNode.operation = new atb_speed::common::MoeDistributeDispatchOperation(
                std::string("MoeDistributeDispatch") + std::to_string(param.packQuantType), dispatchParam);
    }

    moeDistributeDispatchNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"),
                                   GetTensorIdx(tensorMap, "in_selected_experts"),
                                   GetTensorIdx(tensorMap, "in_expert_weight"),
                                   GetTensorIdx(tensorMap, "in_padding_idx")};

    moeDistributeDispatchNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates"),
                                    GetTensorIdx(tensorMap, "intermediate_gmm0_deqscale"),
                                    GetTensorIdx(tensorMap, "intermediate_idx"),
                                    GetTensorIdx(tensorMap, "intermediate_group_list"),
                                    GetTensorIdx(tensorMap, "intermediate_ep_recv_counts"),
                                    GetTensorIdx(tensorMap, "intermediate_tp_recv_counts"),
                                    GetTensorIdx(tensorMap, "intermediate_expand_expert_weight"),
                                    };
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(moeDistributeDispatchNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    ATB_SPEED_LOG_DEBUG("MoeDistributeDispatch calculation success");
    return atb::NO_ERROR;
}

template<typename CombineParam>
atb::Status SetMoeDistributeCombineParam(CombineParam &combineParam, const MoeMlpParam &param)
{
    combineParam.epRankId = param.moeEpRank;
    combineParam.epRankSize = param.moeEpSize;
    combineParam.epCommName = param.moeEpDomain;
    combineParam.moeExpertNum = param.numOfExperts;
    combineParam.localMoeExpertNum = param.numOfDeviceExperts;
    combineParam.maxDecodeDpTokenSize = param.maxDecodeDpTokenSize;
    combineParam.sharedExpertRankNum = param.numDanglingSharedExperts;
    combineParam.topk = param.topk;
    const char *hcclIntraPcie = std::getenv("HCCL_INTRA_PCIE_ENABLE");
    const char *hcclIntraRoce = std::getenv("HCCL_INTRA_ROCE_ENABLE");
    if (hcclIntraPcie != nullptr && hcclIntraRoce != nullptr
        && std::string(hcclIntraPcie) == "1" && std::string(hcclIntraRoce) == "0") {
        combineParam.commQuantMode = 2; // 2: translated2
    }
    return atb::NO_ERROR;
}

atb::Status CreateMoeDistributeCombine(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph)
{
    atb::Node moeDistributeCombineNode;
    if (param.enableDispatchCombineV2) {
        atb_speed::common::MoeDistributeCombineV2Param combineParam;
        SetMoeDistributeCombineParam(combineParam, param);
        moeDistributeCombineNode.operation = \
            new atb_speed::common::MoeDistributeCombineV2Operation("MoeDistributeCombineV2", combineParam);
    } else {
        atb_speed::common::MoeDistributeCombineParam combineParam;
        SetMoeDistributeCombineParam(combineParam, param);
        moeDistributeCombineNode.operation = \
            new atb_speed::common::MoeDistributeCombineOperation("MoeDistributeCombine", combineParam);
    }

    moeDistributeCombineNode.outTensorIds = {GetTensorIdx(tensorMap, "out_moe_mlp_result")};
    moeDistributeCombineNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out"),
                                 GetTensorIdx(tensorMap, "in_selected_experts"),
                                 GetTensorIdx(tensorMap, "intermediate_idx"),
                                 GetTensorIdx(tensorMap, "intermediate_ep_recv_counts"),
                                 GetTensorIdx(tensorMap, "in_expert_weight"),
                                 GetTensorIdx(tensorMap, "intermediate_tp_recv_counts"),
                                 GetTensorIdx(tensorMap, "intermediate_expand_expert_weight"),
                                 };

    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(moeDistributeCombineNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    ATB_SPEED_LOG_DEBUG("MoeDistributeDispatch calculation success");
    return atb::NO_ERROR;
}

// Op5 - Gather1
atb::Status CreateGather1(std::map<std::string, uint32_t> &tensorMap,
    std::shared_ptr<int64_t> batchDimPtr, atb::GraphParam &opGraph)
{
    atb::Node gatherNode1;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode1.operation));
    gatherNode1.inTensorIds = {GetTensorIdx(tensorMap, "in_expert_weight"),
                               GetTensorIdx(tensorMap, "intermediate_weight_idx")};
    gatherNode1.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_weight")};
    gatherNode1.inTensorReshapeFuncs.resize(gatherNode1.inTensorIds.size());
    gatherNode1.inTensorReshapeFuncs[0] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    opGraph.nodes.push_back(gatherNode1);
    ATB_SPEED_LOG_DEBUG("Gather1 calculation success");
    return atb::NO_ERROR;
}

// Op6 - ElewiseMul1
atb::Status CreateElewiseMul1(std::map<std::string, uint32_t> &tensorMap,
    std::shared_ptr<int64_t> batchDimPtr, atb::GraphParam &opGraph)
{
    atb::Node weightMulNode;
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(elewiseParam, &weightMulNode.operation));
    weightMulNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out"),
                                 GetTensorIdx(tensorMap, "intermediate_sorted_weight")};
    weightMulNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out_weighted")};
    weightMulNode.inTensorReshapeFuncs.resize(weightMulNode.inTensorIds.size());
    weightMulNode.inTensorReshapeFuncs[1] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2:number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = 1;
    };
    opGraph.nodes.push_back(weightMulNode);
    ATB_SPEED_LOG_DEBUG("ElewiseMul1 calculation success");
    return atb::NO_ERROR;
}

// Op7 - Argsort
atb::Status CreateArgsort(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node argsortNode;
    atb::infer::GatingParam gatingParam;
    gatingParam.topkExpertNum = 1;
    gatingParam.cumSumNum = 0;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatingParam, &argsortNode.operation));
    argsortNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_weight_idx"),
                               GetTensorIdx(tensorMap, "in_expert_array")};
    argsortNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_dummy_zero"),
                                GetTensorIdx(tensorMap, "intermediate_dummy_one"),
                                GetTensorIdx(tensorMap, "intermediate_rev_idx")};
    opGraph.nodes.push_back(argsortNode);
    ATB_SPEED_LOG_DEBUG("Argsort calculation success");
    return atb::NO_ERROR;
}

// Op8 - Gather2
atb::Status CreateGather2(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node gatherNode2;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode2.operation));
    gatherNode2.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out_weighted"),
                               GetTensorIdx(tensorMap, "intermediate_rev_idx")};
    gatherNode2.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_rev_sorted_hiddenstates")};
    opGraph.nodes.push_back(gatherNode2);
    ATB_SPEED_LOG_DEBUG("Cather2 calculation success");
    return atb::NO_ERROR;
}

// Op9 - Reduction
atb::Status CreateReduction(std::map<std::string, uint32_t> &tensorMap,
    const MoeMlpParam &param, std::shared_ptr<int64_t> batchDimPtr, atb::GraphParam &opGraph)
{
    CHECK_PARAM_NE(param.topk, 0);
    atb::Node reduceNode;
    atb::infer::ReduceParam reduceParam;
    reduceParam.reduceType = atb::infer::ReduceParam::ReduceType::REDUCE_SUM;
    reduceParam.axis = {1};
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(reduceParam, &reduceNode.operation));
    reduceNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_rev_sorted_hiddenstates")};
    reduceNode.outTensorIds = {GetTensorIdx(tensorMap, "out_moe_mlp_result")};
    reduceNode.inTensorReshapeFuncs.resize(reduceNode.inTensorIds.size());
    reduceNode.inTensorReshapeFuncs[0] = [batchDimPtr, param](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3:number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0] / param.topk;
        newShape.dims[1] = param.topk;
        newShape.dims[2] = oldShape.dims[1]; // 2:the third dimension of the new shape
    };
    opGraph.nodes.push_back(reduceNode);
    ATB_SPEED_LOG_DEBUG("Reduction calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateActivationBlock(std::map<std::string, uint32_t> &tensorMap,
    const MoeMlpParam &param, atb::GraphParam &opGraph)
{
    if (param.supportSwiGLU) {
        if (param.enableSwigluQuant) {
            CHECK_OPERATION_STATUS_RETURN(CreateActivationQuant(tensorMap, opGraph));
        } else if (param.swigluBackend == atb_speed::common::OpBackend::ACLNN) {
	    CHECK_OPERATION_STATUS_RETURN(CreateAclnnSwiglu(tensorMap, opGraph));
	} else if (param.swigluBackend == atb_speed::common::OpBackend::ATB){
            CHECK_OPERATION_STATUS_RETURN(CreateActivation(tensorMap, opGraph));
        }
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateSplit(tensorMap, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateActivationO(tensorMap, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateElewiseMul(tensorMap, opGraph));
    }

    ATB_SPEED_LOG_DEBUG("ActivationBlock calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateRecord(const MoeMlpParam &param, atb::GraphParam &opGraph,
                         atb_speed::EventAction eventAction, const std::string &cvKey)
{
    if (param.enableCVOverlap) {
        atb::Node recordNode;
        recordNode.inTensorIds = {};
        recordNode.outTensorIds = {};
        CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().RecordEvent(
            recordNode.operation,
            eventAction,
            cvKey));
        opGraph.nodes.push_back(recordNode);
        ATB_SPEED_LOG_DEBUG("Record event success");
    }
    return atb::NO_ERROR;
}

atb::Status CreateWait(const MoeMlpParam &param, atb::GraphParam &opGraph,
                       atb_speed::EventAction eventAction, const std::string &cvKey)
{
    if (param.enableCVOverlap) {
        atb::Node waitNode;
        waitNode.inTensorIds = {};
        waitNode.outTensorIds = {};
        CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().WaitEvent(
            waitNode.operation,
            eventAction,
            cvKey));
        opGraph.nodes.push_back(waitNode);
        ATB_SPEED_LOG_DEBUG("Wait event success");
    }
    return atb::NO_ERROR;
}

atb::Status CreateMoeDistribute(
    std::map<std::string, uint32_t> &tensorMap,
    const MoeMlpParam &param,
    atb::GraphParam &opGraph)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);
    bool isGMMSwigluQuant = IsGMMSwigluQuant(CalcUpGmmQuantType(param), param);
    CHECK_OPERATION_STATUS_RETURN(CreateMoeDistributeDispatch(tensorMap, param, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateGmm(tensorMap, opGraph, param));
    if (!isGMMSwigluQuant) {
        CHECK_OPERATION_STATUS_RETURN(CreateActivationBlock(tensorMap, param, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateGmm1(tensorMap, opGraph, param));
    }
    CHECK_OPERATION_STATUS_RETURN(CreateMoeDistributeCombine(tensorMap, param, opGraph));
    return atb::NO_ERROR;
}

atb::Status CreateFusedRouting(
    std::map<std::string, uint32_t> &tensorMap,
    const MoeMlpParam &param,
    atb::GraphParam &opGraph)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);
    bool isGMMSwigluQuant = IsGMMSwigluQuant(CalcUpGmmQuantType(param), param);
    if (param.enableCVOverlap) {
        CHECK_OPERATION_STATUS_RETURN(CreateWait(
            param, opGraph, atb_speed::EventAction::POP, atb_speed::common::CUBE_CONTROL));
    }
    if (param.enableInitQuant) {
        CHECK_OPERATION_STATUS_RETURN(CreateInitRoutingQuant(tensorMap, param, opGraph)); //V2/V3-quant
    } else if (param.enableInitRoutingV3) {
	CHECK_OPERATION_STATUS_RETURN(CreateInitRoutingV3NoQuant(tensorMap, param, opGraph)); // V3-noquant
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateInitRouting(tensorMap, param, opGraph)); //v2-noquant
    }
    if (param.hasMoeEp) {
        CHECK_OPERATION_STATUS_RETURN(CreateComputeExpertSlice(tensorMap, param, opGraph));
    }
    if (param.enableInitRoutingCutoff) {
        CHECK_OPERATION_STATUS_RETURN(CreateExpandedXLen(tensorMap, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateGroupListFilter(tensorMap, param, opGraph));
    }

    if (param.enableInitRoutingV3){
        // no need cast;
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateCast(tensorMap, param, opGraph));
    }

    if (param.enableInitQuant){
	// quant + init routing v3
	CHECK_OPERATION_STATUS_RETURN(CreateGmm(tensorMap, opGraph, param));
    } else if (param.enableInitRoutingV3) {
	// no quant + init routing v3
	CHECK_OPERATION_STATUS_RETURN(CreateCustomGmmWeightNZ(tensorMap, opGraph, param));
    } else{
	// no quant
	CHECK_OPERATION_STATUS_RETURN(CreateGmm(tensorMap, opGraph, param));
    }

    if (param.enableCVOverlap) {
        CHECK_OPERATION_STATUS_RETURN(CreateWait(
            param, opGraph, atb_speed::EventAction::POP, atb_speed::common::VECTOR_CONTROL));
        CHECK_OPERATION_STATUS_RETURN(CreateRecord(
            param, opGraph, atb_speed::EventAction::POP, atb_speed::common::CUBE_CONTROL));
    }

    if (!isGMMSwigluQuant){
	if (!param.enableInitQuant && param.enableInitRoutingV3){
            // no-quant && initRoutingV3
            CHECK_OPERATION_STATUS_RETURN(CreateActivationBlock(tensorMap, param, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateCustomGmm1WeightNZ(tensorMap, opGraph, param));
    	} else {
            // quant && initRoutingV3
            // default
            CHECK_OPERATION_STATUS_RETURN(CreateActivationBlock(tensorMap, param, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateGmm1(tensorMap, opGraph, param));
    	}
    }

    CHECK_OPERATION_STATUS_RETURN(CreateMoeTokenUnpermute(tensorMap, param, opGraph));
    if (param.enableCVOverlap) {
        CHECK_OPERATION_STATUS_RETURN(CreateWait(
            param, opGraph, atb_speed::EventAction::POP, atb_speed::common::CUBE_CONTROL));
    }
    return atb::NO_ERROR;
}

atb::Status CreateDefault(
    std::map<std::string, uint32_t> &tensorMap,
    const MoeMlpParam &param,
    atb::GraphParam &opGraph)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);
    bool isGMMSwigluQuant = IsGMMSwigluQuant(CalcUpGmmQuantType(param), param);
    CHECK_OPERATION_STATUS_RETURN(CreateGating(tensorMap, param, batchDimPtr, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateGather0(tensorMap, opGraph));
    if (param.packQuantType != atb_speed::common::PackQuantType::ALL_FP && Is310P()) {
        // 310P QuantGMMDequantOperation
        CHECK_OPERATION_STATUS_RETURN(CreateQuantGMMDequant(tensorMap, opGraph, param));
        CHECK_OPERATION_STATUS_RETURN(CreateActivationBlock(tensorMap, param, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateQuantGMMDequant1(tensorMap, opGraph, param));
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateGmm(tensorMap, opGraph, param));
        if (!isGMMSwigluQuant) {
            CHECK_OPERATION_STATUS_RETURN(CreateActivationBlock(tensorMap, param, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateGmm1(tensorMap, opGraph, param));
        }
    }
    CHECK_OPERATION_STATUS_RETURN(CreateGather1(tensorMap, batchDimPtr, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateElewiseMul1(tensorMap, batchDimPtr, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateArgsort(tensorMap, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateGather2(tensorMap, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateReduction(tensorMap, param, batchDimPtr, opGraph));
    return atb::NO_ERROR;
}

atb::Status CreateMoeMlpOperation(const MoeMlpParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "MoeMlp";
    
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.inTensorNum " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.outTensorNum " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.internalTensorNum" << opGraph.internalTensorNum);

    if (param.hasMoeEp && !param.enableMoeDistribute) {
        CHECK_OPERATION_STATUS_RETURN(CreateExpertFilter(tensorMap, param, opGraph));
    }
    if (param.enableMoeDistribute) {
        CHECK_OPERATION_STATUS_RETURN(CreateMoeDistribute(tensorMap, param, opGraph));
    } else if (param.enableFusedRouting) {
        CHECK_OPERATION_STATUS_RETURN(CreateFusedRouting(tensorMap, param, opGraph));
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateDefault(tensorMap, param, opGraph));
    }

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (param.enableExpertCumSumOutput) {
            outTensorDescs.at(1) = atb::TensorDesc{};
            outTensorDescs.at(1).format = ACL_FORMAT_ND;
            outTensorDescs.at(1).shape.dimNum = 1;
            outTensorDescs.at(1).dtype = ACL_INT64;
            outTensorDescs.at(1).shape.dims[0] = param.numOfDeviceExperts;
        }

	if (param.enableInitRoutingV3 && !param.enableInitQuant && inTensorDescs.at(1).format != ACL_FORMAT_FRACTAL_NZ) {
	  ATB_SPEED_LOG_ERROR("For non-quantized cases, enabing init_routing_v3 uses moe_grouped_matmul by default and require in_expert_weight need FRACTAL_NZ format");
          return atb::ERROR_INVALID_TENSOR_FORMAT;
	}

        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
}
} // namespace atb_speed
