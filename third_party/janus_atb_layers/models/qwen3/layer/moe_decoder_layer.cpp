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

#include "operations/aclnn/ops/rms_norm_operation.h"
#include "operations/aclnn/ops/add_rms_norm_operation.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/attention/fusion_attention.h"
#include "operations/fusion/mlp/mlp.h"
#include "operations/fusion/moe/sparse_moe.h"
#include "models/qwen3/operation/qwen_mlp_shared_expert.h"
#include "models/qwen3/layer/moe_decoder_layer.h"

namespace atb_speed {
namespace qwen {
static const uint64_t IN_TENSOR_COUNT = 69;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 5;
static const uint64_t NODE_COUNT = 8;

std::map<std::string, std::vector<std::string>> GetQwenMoeLayerInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qwenMoeLayerInTensorCandidates = {
        {"default", {
            "in_hidden_states", "in_expert_array", "in_expert_group", "in_one_hot", "in_zero_hot",
            "in_cos_table", "in_sin_table", "in_attention_mask", "in_k_cache", "in_v_cache",
            "in_seq_len", "in_place_holder", "in_token_offset", "in_layer_id", "in_block_tables", "in_slots"}},
        {"default_weight", {
            "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias",
            "in_qkv_weight_0", "in_qkv_bias_0", "in_qkv_descale_0",
            "in_qkv_offset_0", "in_qkv_scale_0", "in_qkv_compress_idx_0",
            "in_qkv_weight_1", "in_qkv_bias_1", "in_qkv_descale_1", "in_qkv_offset_1",
            "in_qkv_scale_1", "in_qkv_compress_idx_1",
            "in_qkv_weight_2", "in_qkv_bias_2", "in_qkv_descale_2", "in_qkv_offset_2",
            "in_qkv_scale_2", "in_qkv_compress_idx_2",
            "in_attention_out_weight", "in_attention_out_bias", "in_attention_out_descale",
            "in_attention_out_offset", "in_attention_out_scale", "in_attention_out_compress_idx",
            "in_q_norm_weight", "in_k_norm_weight",
            "in_selfattention_out_norm_weight", "in_selfattention_out_norm_bias",
            "in_selfattention_out_new_norm_weight", "in_selfattention_out_new_norm_bias",
            "in_block_sparse_moe_gate_weight", "in_block_sparse_moe_gate_bias",
            "in_block_sparse_moe_gate_descale", "in_block_sparse_moe_gate_offset",
            "in_block_sparse_moe_gate_scale", "in_block_sparse_moe_gate_compress_idx",
            "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert", "in_mlp_gateup_descale_expert",
            "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert", "in_mlp_gateup_compress_idx_expert",
            "in_mlp_down_weight_expert", "in_mlp_down_bias_expert", "in_mlp_down_descale_expert",
            "in_mlp_down_offset_expert", "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert",
            "in_mlp_shared_gateup_weight", "in_mlp_shared_down_weight", "in_mlp_shared_expert_gate"
        }},
        {"q_len", {"in_q_len"}},
        {"acl_graph", {"paged_attention_tiling_data"}},
        {"externel_add_norm", {"in_residual"}},
        {"quant_externel_add_norm", {"quant_add_norm_scaling", "quant_add_norm_offset"}} 
    };
    return qwenMoeLayerInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetQwenMoeLayerIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qwenMoeLayerIntermediateTensorCandidates = {
        {"default", {
            "intermediate_attention_out", "intermediate_selfattention_norm_out", "intermediate_mlp_out"
        }},
        {"default_add_norm", {
            "intermediate_selfattention_norm_out", "intermediate_add_out"
        }},
        {"moe", {
            "intermediate_moe_out"
        }},
        {"shared_expert", {
            "intermediate_shared_experts_out"
        }},
        {"useless", {"intermediate_rstd"}}
    };
    return qwenMoeLayerIntermediateTensorCandidates;
}

std::map<std::string, uint32_t> QwenMoeConstructTensorMap(
    const MoeDecoderLayerParam &param, uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto qwenMoeLayerInTensorCandidates = GetQwenMoeLayerInTensorCandidates();
    auto qwenMoeLayerIntermediateTensorCandidates = GetQwenMoeLayerIntermediateTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out_decoder_layer"};

    atb_speed::common::AddTensorToList(qwenMoeLayerInTensorCandidates, "default_weight", inTensorList);
    atb_speed::common::AddTensorToList(qwenMoeLayerInTensorCandidates, "default", inTensorList);

    if (param.enableSpeculate || param.enableSplitFuse) {
        atb_speed::common::AddTensorToList(qwenMoeLayerInTensorCandidates, "q_len", inTensorList);
    }
    if (param.enableAclGraphPagedAttention && !param.isPrefill) {
        atb_speed::common::AddTensorToList(qwenMoeLayerInTensorCandidates, "acl_graph", inTensorList);
    }
    if (param.enableAclnnExternelAddRmsNorm){
    	atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates, "default_add_norm", intermediateTensorList);
    } else{
	atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates, "default", intermediateTensorList);
    }
    if (param.hasMoe) {
        atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates, "moe", intermediateTensorList);
    }
    if (param.hasSharedExpert) {
        atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates,
                                           "shared_expert", intermediateTensorList);
    }

    if (param.enableAclnnRmsNorm || param.enableAclnnAddRmsNorm) {
        atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates,
                                           "useless", intermediateTensorList);
    }

    if (param.enableAclnnExternelAddRmsNorm){
        atb_speed::common::AddTensorToList(qwenMoeLayerInTensorCandidates, "externel_add_norm", inTensorList);
        atb_speed::common::AddTensorToList(qwenMoeLayerInTensorCandidates, "quant_externel_add_norm", inTensorList);
        outTensorList = {"intermediate_mlp_out", "intermediate_attention_out"};
    }
    inTensorNum = inTensorList.size();
    internalTensorNum = intermediateTensorList.size();
    outTensorNum = outTensorList.size();
    ATB_SPEED_LOG_DEBUG("ConstructTensorMap done");
    std::stringstream ss;
    int count = 0;
    return atb_speed::common::GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

atb::Status SetFusionAttentionParam(
    const MoeDecoderLayerParam &param,
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam
)
{
    // QKV linear param
    fusionAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.isBF16 = param.isBF16;
    fusionAttentionParam.layerLinearQuantType = param.attnLinearQuantType;
    fusionAttentionParam.layerLinearTransposeType = param.attnLinearTransposeType;
    fusionAttentionParam.packQuantType = param.packQuantType.at(0);
    fusionAttentionParam.supportLcoc = param.enableLcoc;
    fusionAttentionParam.enableSplitFuse = param.enableSplitFuse;
    atb::infer::RmsNormParam attenRmsNormParam;
    attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormParam.normParam.epsilon = param.normEps;
    fusionAttentionParam.normParamType = attenRmsNormParam;
    atb::infer::RmsNormParam attenRmsNormQuantParam;
    attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormQuantParam.normParam.epsilon = param.normEps;
    attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    fusionAttentionParam.normQuantParamType = attenRmsNormQuantParam;

    fusionAttentionParam.enableAclnnRmsNorm = param.enableAclnnRmsNorm;
    fusionAttentionParam.enableAclGraphPagedAttention = param.enableAclGraphPagedAttention;
    fusionAttentionParam.enableAddNorm = param.enableAclnnExternelAddRmsNorm;
    fusionAttentionParam.matmulBackend = atb_speed::common::OpBackend::ACLNN;
    return atb::NO_ERROR;
}

atb::Status SetFusionAttentionNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph, size_t &nodeId,
                                   std::map<std::string, uint32_t> &tensorMap)
{
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    SetFusionAttentionParam(param, fusionAttentionParam);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    // rope param
    fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::ALL_ROTARY;
    fusionAttentionParam.ropeParam.rotaryCoeff = 2; // 2:translated
    // self attention param
    fusionAttentionParam.isFA = param.isFA;
    fusionAttentionParam.isPrefill = param.isPrefill;
    fusionAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    fusionAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    CHECK_PARAM_GT(param.hiddenSizePerAttentionHead, 0);
    fusionAttentionParam.selfAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    fusionAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
    if (param.isFA) {
        fusionAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    fusionAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    if (param.enableSpeculate || param.enableSplitFuse) {
        fusionAttentionParam.pageAttentionParam.calcType = \
            atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
        fusionAttentionParam.pageAttentionParam.maskType = \
            atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC;
    } else {
        fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::UNDEFINED;
    }
    fusionAttentionParam.useQKNorm = param.useQKNorm;
    fusionAttentionParam.qkvHasBias = param.linearHasBias.at(atb_speed::base::QKV_HASBIAS);
    fusionAttentionParam.selfOutLinearTensorParallelInfo = {
        param.rank, param.worldSize, param.backend, param.rankTableFile
    };
    if (param.mapping.isInitialized_) {
	atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
        parallelInfo.InitCommDomain(
	    fusionAttentionParam.selfOutLinearTensorParallelInfo.hcommInfo,
	    fusionAttentionParam.selfOutLinearTensorParallelInfo.commDomain);
    }
    CHECK_OPERATION_STATUS_RETURN(Attention(fusionAttentionParam, &attentionNode.operation));
    std::vector<std::string> attnInTensorNames = {
            "in_hidden_states", "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight",
            "in_input_norm_new_bias", "in_qkv_weight_0", "in_qkv_scale_0", "in_qkv_offset_0",
            "in_qkv_descale_0", "in_qkv_bias_0", "in_qkv_compress_idx_0",
            "in_qkv_weight_1", "in_qkv_scale_1", "in_qkv_offset_1", "in_qkv_descale_1",
            "in_qkv_bias_1", "in_qkv_compress_idx_1", "in_qkv_weight_2", "in_qkv_scale_2",
            "in_qkv_offset_2", "in_qkv_descale_2", "in_qkv_bias_2", "in_qkv_compress_idx_2",
            "in_cos_table", "in_sin_table", "in_seq_len", "in_k_cache", "in_v_cache",
            "in_attention_mask", "in_token_offset", "in_layer_id", "in_block_tables", "in_slots",
            "in_attention_out_weight", "in_attention_out_scale", "in_attention_out_offset",
            "in_attention_out_descale", "in_attention_out_bias", "in_attention_out_compress_idx"
            };
    if (param.enableAclnnExternelAddRmsNorm){
        attnInTensorNames.push_back("quant_add_norm_scaling");
        attnInTensorNames.push_back("quant_add_norm_offset");
    }
    if (param.enableSpeculate || param.enableSplitFuse) {
        attnInTensorNames.push_back("in_q_len");
    }
    if (fusionAttentionParam.useQKNorm) {
        attnInTensorNames.push_back("in_q_norm_weight");
        attnInTensorNames.push_back("in_k_norm_weight");
    }

    std::vector<std::string> attnOutTensorNames = {"intermediate_attention_out"};
    if (param.enableAclnnExternelAddRmsNorm){
        attnInTensorNames.push_back("in_residual");
        attnOutTensorNames.push_back("intermediate_add_out");
    } 

    if (fusionAttentionParam.enableAclGraphPagedAttention && !param.isPrefill) {
        attnInTensorNames.push_back("paged_attention_tiling_data");
    }

    attentionNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, attnInTensorNames);
    attentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, attnOutTensorNames);
    return atb::NO_ERROR;
}


atb::Status SetAttentionResidualAddNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph, size_t &nodeId,
                                        std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    std::string hiddenAttnOutStr = param.enableAclnnExternelAddRmsNorm? "intermediate_add_out" : "in_hidden_states"; 
    selfResidualAddNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {hiddenAttnOutStr, "intermediate_attention_out"});
    selfResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out"});
    return atb::NO_ERROR;
}

atb::Status SetSelfNormNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph,
                            size_t &nodeId, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    if (param.enableAclnnRmsNorm) {
        selfNormNode.operation = new atb_speed::common::RmsNormOperation("RmsNorm", param.normEps);
    } else {
        atb::infer::RmsNormParam selfNormParam;
        selfNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        selfNormParam.normParam.epsilon = param.normEps;
        CreateOperation(selfNormParam, &selfNormNode.operation);
    }
    if (selfNormNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("selfNormNode op is nullptr: ");
    }
    selfNormNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out",
                                                        "in_selfattention_out_norm_weight"});
    selfNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_selfattention_norm_out"});
    if (param.enableAclnnRmsNorm) {
        selfNormNode.outTensorIds.push_back(atb_speed::common::GetTensorIdx(tensorMap, "intermediate_rstd"));
    }
    ATB_SPEED_LOG_DEBUG("create post normEps");
    return atb::NO_ERROR;
}

atb::Status SetAddNormNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph,
                            size_t &nodeId, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node &selfAddNormNode = opGraph.nodes.at(nodeId++);
    selfAddNormNode.operation = new atb_speed::common::AddRmsNormOperation("AddRmsNormNode", 1e-6);
    std::string hiddenAttnOutStr = param.enableAclnnExternelAddRmsNorm? "intermediate_add_out" : "in_hidden_states"; 
    selfAddNormNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {hiddenAttnOutStr,
                                                        "intermediate_attention_out",
                                                        "in_selfattention_out_norm_weight"});
        
    selfAddNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {
        "intermediate_selfattention_norm_out",
        "intermediate_rstd",
        "intermediate_attention_out"});
    ATB_SPEED_LOG_DEBUG("create SetAttentionResidualAddNormNode");
    return atb::NO_ERROR;
}

atb::Status SetMoeNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph,
                       size_t &nodeId, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node &moeNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::SparseMoeParam sparseMoeParam;
    sparseMoeParam.transpose = param.transpose;
    sparseMoeParam.numOfExperts = param.numOfExperts;
    sparseMoeParam.num = param.numOfSelectedExperts;
    sparseMoeParam.isBF16 = param.isBF16;
    sparseMoeParam.expertParallelDegree = param.expertParallelDegree;
    sparseMoeParam.processLogits = param.processLogits;
    sparseMoeParam.supportSwiGLU = param.enableSwiGLU;
    sparseMoeParam.routingMethod = param.routingMethod;
    sparseMoeParam.enableFusedRouting = param.enableFusedRouting;
    sparseMoeParam.enableInitRoutingV3 = param.enableInitRoutingV3;
    sparseMoeParam.enableFusedReducesumDiv = param.enableFusedReducesumDiv;
    sparseMoeParam.swigluBackend = param.swigluBackend;
    sparseMoeParam.moeLinearQuantType = param.moeLinearQuantType;
    sparseMoeParam.gateUpTransposeB = param.moeLinearTransposeType[atb_speed::common::SparseMoeIdx::MOE_MLP_GATE_IDX];
    sparseMoeParam.downTransposeB = param.moeLinearTransposeType[atb_speed::common::SparseMoeIdx::MOE_MLP_DOWN_IDX];
    sparseMoeParam.packQuantType = param.packQuantType.at(1);
    atb_speed::common::CreateSparseMoeOperation(sparseMoeParam, &moeNode.operation);
    if (moeNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("SparseMoe op is nullptr: ");
    }
    std::vector<std::string> moeInTensorNames = {
        "intermediate_selfattention_norm_out", "in_block_sparse_moe_gate_weight",
        "in_block_sparse_moe_gate_bias", "in_block_sparse_moe_gate_descale",
        "in_block_sparse_moe_gate_offset", "in_block_sparse_moe_gate_scale",
        "in_block_sparse_moe_gate_compress_idx", "in_mlp_gateup_weight_expert",
        "in_mlp_gateup_bias_expert", "in_mlp_gateup_descale_expert",
        "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert",
        "in_mlp_gateup_compress_idx_expert", "in_mlp_down_weight_expert",
        "in_mlp_down_bias_expert", "in_mlp_down_descale_expert", "in_mlp_down_offset_expert",
        "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert", "in_expert_array",
        "in_expert_group", "in_one_hot", "in_zero_hot"};
    moeNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, moeInTensorNames);
    moeNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out"});

    ATB_SPEED_LOG_DEBUG("Moe Dense calculation success");
    return atb::NO_ERROR;
}

atb::Status SetShareExpertNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph, size_t &nodeId,
                               std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node &shareExpertNode = opGraph.nodes.at(nodeId++);
    atb_speed::qwen::QwenMlpSharedExpertParam sharedMlpExpertParam;
    sharedMlpExpertParam.transpose = param.transpose;
    sharedMlpExpertParam.hasSharedExpertGate = param.hasSharedExpertGate;
    ATB_SPEED_LOG_DEBUG("sharedMlpExpertParam success");
    qwen::CreateQwenMlpSharedExpertOperation(
        sharedMlpExpertParam, &shareExpertNode.operation);
    shareExpertNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_selfattention_norm_out",
                                                        "in_mlp_shared_gateup_weight",
                                                        "in_mlp_shared_down_weight", "in_mlp_shared_expert_gate"});
    shareExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_shared_experts_out"});
    ATB_SPEED_LOG_DEBUG("shared expert calculation success");
    return atb::NO_ERROR;
}

atb::Status SetShareAddSelectNode(atb::GraphParam &opGraph, size_t &nodeId, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node &shareAddSelectNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &shareAddSelectNode.operation);
    shareAddSelectNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_shared_experts_out", "intermediate_moe_out"});
    shareAddSelectNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out"});
    ATB_SPEED_LOG_DEBUG("shared expert add success");
    
    return atb::NO_ERROR;
}

atb::Status SetAllReduceNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph, size_t &nodeId,
                             std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node &moeAllReduceNode = opGraph.nodes.at(nodeId++);
    atb::infer::AllReduceParam allReduceParam;
    
    atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::MLP_TP);
    allReduceParam.rank = parallelInfo.rank;
    allReduceParam.rankSize = parallelInfo.rankIds.size();
    allReduceParam.backend = parallelInfo.defaultBackend;
    if (param.mapping.isInitialized_) {
        parallelInfo.InitCommDomain(allReduceParam.hcclComm, allReduceParam.commDomain);
    }
    CreateOperation(allReduceParam, &moeAllReduceNode.operation);
    if (moeAllReduceNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("moeAllReduceNode op is nullptr: ");
    }
    moeAllReduceNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap,
            {param.hasMoe ? "intermediate_moe_out" : "intermediate_shared_experts_out"});
    moeAllReduceNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out"});
    ATB_SPEED_LOG_DEBUG("create all reduce");
    return atb::NO_ERROR;
}

atb::Status SetMlpResidualAddNode(atb::GraphParam &opGraph, size_t &nodeId, atb::Operation **operation,
                                  std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    
    mlpResidualAddNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out", "intermediate_mlp_out"});
    mlpResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"out_decoder_layer"});

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    ATB_SPEED_LOG_DEBUG("decoder layer: residule create opgraph");
    return atb::NO_ERROR;
}

atb::Status MoeDecoderLayer(const MoeDecoderLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";
    size_t nodeId = 0;
    std::map<std::string, uint32_t> tensorMap = QwenMoeConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    
    int32_t node_count = NODE_COUNT;
    if (param.enableAclnnExternelAddRmsNorm)
        node_count -= 1;
    if (param.enableAclnnAddRmsNorm)
        node_count -= 1;

    if (!param.hasMoe || !param.hasSharedExpert) {
        opGraph.nodes.resize(node_count - 2); // translated,translated2translated
    } else {
        opGraph.nodes.resize(node_count);
    }

    // node0: attention
    CHECK_OPERATION_STATUS_RETURN(SetFusionAttentionNode(param, opGraph, nodeId, tensorMap));
    if (param.enableAclnnAddRmsNorm){
        CHECK_OPERATION_STATUS_RETURN(SetAddNormNode(param, opGraph, nodeId, tensorMap));
    } else {
        // node1: residual
        CHECK_OPERATION_STATUS_RETURN(SetAttentionResidualAddNode(param, opGraph, nodeId, tensorMap));
        // node2: norm
        CHECK_OPERATION_STATUS_RETURN(SetSelfNormNode(param, opGraph, nodeId, tensorMap));
    }

    // node3: moe
    if (param.hasMoe) {
        CHECK_OPERATION_STATUS_RETURN(SetMoeNode(param, opGraph, nodeId, tensorMap));
    }
    // node4: shareExpert
    if (param.hasSharedExpert) {
        CHECK_OPERATION_STATUS_RETURN(SetShareExpertNode(param, opGraph, nodeId, tensorMap));
    }
    // node5: shareExperts add moe
    if (param.hasMoe && param.hasSharedExpert) {
        CHECK_OPERATION_STATUS_RETURN(SetShareAddSelectNode(opGraph, nodeId, tensorMap));
    }
    // node6: addreduce
    CHECK_OPERATION_STATUS_RETURN(SetAllReduceNode(param, opGraph, nodeId, tensorMap));
    if(!param.enableAclnnExternelAddRmsNorm) {
        // node7: residual
        CHECK_OPERATION_STATUS_RETURN(SetMlpResidualAddNode(opGraph, nodeId, operation, tensorMap));
    }
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

MoeDecoderLayer::MoeDecoderLayer() {}
MoeDecoderLayer::~MoeDecoderLayer() {}
}  // namespace qwen
}  // namespace atb_speed
