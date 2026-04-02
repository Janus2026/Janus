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

#include "operations/fusion/linear/linear.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/aclnn/ops/add_rms_norm_operation.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace moe {

void MoeLayerParam::PrintParam()
{
    LayerParam::PrintParam();
    std::stringstream ss;
    ss << "Moe Layer Param: " << "enableTopKSoftmax: " << this->enableTopKSoftmax
       << ", transpose: " << this->transpose
       << ", numOfExperts: " << this->numOfExperts
       << ", expertParallelDegree: " << this->expertParallelDegree
       << ", hasSharedExpert: " << this->hasSharedExpert
       << ", hasSharedExpertGate: " << this->hasSharedExpertGate
       << ", numOfGroups: " << this->numOfGroups
       << ", numOfSharedExperts: " << this->numOfSharedExperts
       << ", firstKDenseReplace: " << this->firstKDenseReplace
       << ", topkGroups: " << this->topkGroups;
    ATB_SPEED_LOG_DEBUG(ss.str());
}

atb::Status MoeLayerParam::CalculateDataPartition()
{
    this->enableDynamicEp = this->expertParallelDegree == 2; // 2: dynamic ep level
    this->attnStreamNum = this->mapping.Get(base::ATTN_DP).rankIds.size();
    if (this->layerId < this->firstKDenseReplace) {
        if (this->isMlpFullTP) {
            this->ffnStreamNum = 1;
        } else {
            this->ffnStreamNum = this->mapping.Get(base::ATTN_DP).rankIds.size();
        }
    } else {
        if (this->enableDynamicEp) {
            this->ffnStreamNum = this->mapping.Get(base::MOE_EP).rankIds.size() *
            this->mapping.Get(base::MOE_TP).rankIds.size();
        } else {
            this->ffnStreamNum = 1; // MoE data parallelism is not supported yet
        }
    }
    this->lmheadStreamNum = 1;
    ATB_SPEED_LOG_DEBUG("CalculateDataPartition done"
        << ". Attention Stream Num is " << this->attnStreamNum
        << " . FFN Stream Num is " << this->ffnStreamNum
        << " . lmheadStreamNum Stream Num is " << this->lmheadStreamNum);
    return atb::NO_ERROR;
}

atb::Status MoeLayerParam::CalculateCommType()
{
    if (this->tensorParallelInfo.worldSize == 1) {
        return atb::NO_ERROR;
    }
    int outStreamNum = this->isLastLayer ? this->lmheadStreamNum : this->attnStreamNum;
    this->attnAllreduce = this->mapping.Get(base::ATTN_TP).IsEnabled() && this->ffnStreamNum == this->attnStreamNum;
    this->attnReduceScatter = !this->attnAllreduce && this->mapping.Get(base::ATTN_TP).IsEnabled();
    this->attnAllGather = (this->attnReduceScatter && this->tensorParallelInfo.worldSize > this->ffnStreamNum) || \
                          (this->attnStreamNum > this->ffnStreamNum);
    this->hasAttnComm = this->attnReduceScatter || this->attnAllGather;

    this->ffnAllreduce = this->attnAllreduce && this->ffnStreamNum == outStreamNum;
    this->ffnReduceScatter = !this->ffnAllreduce && uint32_t(this->ffnStreamNum) < this->mapping.worldSize_;
    int ffnOutStreamNum = this->ffnReduceScatter ? this->mapping.worldSize_ : this->ffnStreamNum;
    this->ffnAllGather = ffnOutStreamNum != outStreamNum;
    this->hasFfnComm = this->ffnReduceScatter || this->ffnAllGather;

    ATB_SPEED_LOG_DEBUG("CalculateCommType done"
        << ". outStreamNum is " << outStreamNum
        << ". attnAllreduce is " << this->attnAllreduce << " . attnReduceScatter is " << this->attnReduceScatter
        << " . attnAllGather is " << this->attnAllGather
        << " . ffnAllreduce is " << this->ffnAllreduce << " . ffnReduceScatter is " << this->ffnReduceScatter
        << " . ffnAllGather is " << this->ffnAllGather);
    return atb::NO_ERROR;
}

template <typename NormType>
MoeDecoderLayer<NormType>::MoeDecoderLayer(
    const MoeLayerParam &param) : atb_speed::base::DecoderLayer<NormType>(
        static_cast<atb_speed::base::LayerParam>(param))
{
    this->param = param;
    this->param.CheckParam();
    this->inTensorCandidates["moe_weight"] = {
        "block_sparse_moe_gate_weight", "block_sparse_moe_gate_bias", "block_sparse_moe_gate_descale",
        "block_sparse_moe_gate_offset", "block_sparse_moe_gate_scale", "block_sparse_moe_gate_compress_idx",
        "in_mlp_gateup_weight", "in_mlp_gateup_bias", "in_mlp_gateup_descale", "in_mlp_gateup_offset",
        "in_mlp_gateup_scale", "in_mlp_gateup_compress_idx",
        "in_mlp_down_weight", "in_mlp_down_bias", "in_mlp_down_descale", "in_mlp_down_offset",
        "in_mlp_down_scale", "in_mlp_down_compress_idx"
    };
    this->inTensorCandidates["default_moe"] = {
        "expert_array", "expert_group", "one_hot", "zero_hot"
    };
    this->inTensorCandidates["parallel"] = {
        "in_attn_padding_idx", "in_attn_unpadding_idx", "in_ffn_padding_idx",
        "in_ffn_unpadding_idx", "in_lm_head_skip_padding_token_indices",
        "in_attention_padding_idx_slice", "in_start_expert_idx",
        "in_device_expert_count",
        "in_lty_idx", "in_moe_idx"};
    if (this->param.hasSharedExpert || this->param.isDenseLayer) {
        this->inTensorCandidates["shared_expert_weight"] = {
            "in_mlp_gateup_weight_shared_expert", "in_mlp_gateup_bias_shared_expert",
            "in_mlp_gateup_descale_shared_expert", "in_mlp_gateup_offset_shared_expert",
            "in_mlp_gateup_scale_shared_expert", "in_mlp_gateup_compress_idx_shared_expert",
            "in_mlp_down_weight_shared_expert", "in_mlp_down_bias_shared_expert",
            "in_mlp_down_descale_shared_expert",
            "in_mlp_down_offset_shared_expert", "in_mlp_down_scale_shared_expert",
            "in_mlp_down_compress_idx_shared_expert",
            "in_shared_expert_gate_weight", "in_shared_expert_gate_bias",
            "in_shared_expert_gate_descale",
            "in_shared_expert_gate_offset", "in_shared_expert_gate_scale",
            "in_shared_expert_gate_compress_idx"
        };
    }
    this->internalTensorCandidates["default_moe"] = {"intermediate_norm_out"};
    this->internalTensorCandidates["default_moe_addnorm"] = {"intermediate_norm_rstd_out"};
    if (this->param.layerId >= this->param.firstKDenseReplace && this->param.hasSharedExpert) {
        this->internalTensorCandidates["default_moe"].push_back("intermediate_moe_out");
        this->internalTensorCandidates["default_moe"].push_back("shared_expert_out");
    }
    this->internalTensorCandidates["hiddenstates_padding_slice"] = {
        "intermediate_hidden_states_padding", "intermediate_hidden_states_slice"};
    this->internalTensorCandidates["attn_need_padding"] = {
        "intermediate_attn_out_padding", "intermediate_norm_out_with_padding"};
    this->internalTensorCandidates["attn_reduce_scatter"] = {"intermediate_attn_out_scatter"};
    this->internalTensorCandidates["attn_allgather"] = {"intermediate_norm_out_partial"};
    this->internalTensorCandidates["ffn_need_padding"] = {"intermediate_mlp_out_padding"};
    this->internalTensorCandidates["ffn_reduce_scatter"] = {"intermediate_mlp_out_scatter"};
    this->internalTensorCandidates["ffn_allgather"] = {"intermediate_mlp_out_padding_partial"};
}

template <typename NormType>
void MoeDecoderLayer<NormType>::ConstructInTensorMap()
{
    this->inTensorList.clear();
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "input_norm_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "attn_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "post_attn_norm_weight", this->inTensorList);
    if (this->param.hasSharedExpert || this->param.isDenseLayer) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "shared_expert_weight", this->inTensorList);
    }
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "moe_weight", this->inTensorList);
    if (this->param.useQKNorm) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "qk_norm", this->inTensorList);
    }
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "default", this->inTensorList);
    if (this->param.hasAttnDp) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "attn_dp", this->inTensorList);
    }
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "default_moe", this->inTensorList);
    if (this->param.enableSpeculate || this->param.enableSplitFuse || this->param.enablePrefixCache) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "q_len", this->inTensorList);
    }
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "parallel", this->inTensorList);
    if (this->param.enableAclGraphPagedAttention && !this->param.isPrefill) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "acl_graph", this->inTensorList);
    }
}

template <typename NormType>
void MoeDecoderLayer<NormType>::ConstructInternalTensorMap()
{
    this->intermediateTensorList.clear();
    atb_speed::common::AddTensorToList(this->internalTensorCandidates, "default", this->intermediateTensorList);
    atb_speed::common::AddTensorToList(this->internalTensorCandidates, "default_moe", this->intermediateTensorList);

    if (this->param.enableIntraLayerAddNorm) {
        atb_speed::common::AddTensorToList(this->internalTensorCandidates, "default_moe_addnorm", this->intermediateTensorList);
    }
    if (this->param.hasAttnComm) {
        atb_speed::common::AddTensorToList(
            this->internalTensorCandidates, "attn_need_padding", this->intermediateTensorList);
        atb_speed::common::AddTensorToList(
            this->internalTensorCandidates, "hiddenstates_padding_slice", this->intermediateTensorList);
        if (this->param.attnReduceScatter) {
            atb_speed::common::AddTensorToList(
                this->internalTensorCandidates, "attn_reduce_scatter", this->intermediateTensorList);
        }
        if (this->param.attnAllGather) {
            atb_speed::common::AddTensorToList(
                this->internalTensorCandidates, "attn_allgather", this->intermediateTensorList);
        }
    }
    if (this->param.hasFfnComm) {
        atb_speed::common::AddTensorToList(
            this->internalTensorCandidates, "ffn_need_padding", this->intermediateTensorList);
        if (this->param.ffnReduceScatter) {
            atb_speed::common::AddTensorToList(
                this->internalTensorCandidates, "ffn_reduce_scatter", this->intermediateTensorList);
        }
        if (this->param.ffnAllGather) {
            atb_speed::common::AddTensorToList(
                this->internalTensorCandidates, "ffn_allgather", this->intermediateTensorList);
        }
    }
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddOperationToGraph()
{
    CHECK_OPERATION_STATUS_RETURN(this->AddAttention());
    CHECK_OPERATION_STATUS_RETURN(this->AddPostAttentionProcess());
    CHECK_OPERATION_STATUS_RETURN(this->AddFFN());
    CHECK_OPERATION_STATUS_RETURN(this->AddPostFFNProcess());
    ATB_SPEED_LOG_DEBUG("Add Op to Graph success");
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddAttention()
{
    CHECK_OPERATION_STATUS_RETURN(this->AddFusionAttention(0, 0));
    ATB_SPEED_LOG_DEBUG("Add Attention to Graph success");
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::BuildGraph(atb::Operation **operation)
{

    this->param.enableIntraLayerAddNorm = true;
    this->param.enableFusedTopk = true;

    this->graph.name = this->param.isPrefill ? "Prefill_layer" : "Decoder_layer";
    this->param.CalculateDataPartition();
    this->param.CalculateCommType();
    this->ConstructInTensorMap();
    this->ConstructInternalTensorMap();
    this->graph.inTensorNum = this->inTensorList.size();
    ATB_SPEED_LOG_DEBUG("this->graph.inTensorNum " << this->graph.inTensorNum);
    this->graph.internalTensorNum = this->intermediateTensorList.size();
    ATB_SPEED_LOG_DEBUG("this->graph.internalTensorNum " << this->graph.internalTensorNum);
    this->graph.outTensorNum = this->outTensorList.size();
    ATB_SPEED_LOG_DEBUG("this->graph.outTensorNum " << this->graph.outTensorNum);
    this->tensorMap = atb_speed::common::GetTensorMap(
        this->inTensorList, this->outTensorList, this->intermediateTensorList);
    std::stringstream ss;
    // Add layer map print.
    for (auto tensor = this->tensorMap.cbegin(); tensor != this->tensorMap.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("layer map tensor:\n" << ss.str());

    CHECK_OPERATION_STATUS_RETURN(this->AddOperationToGraph());
    
    uint32_t inHiddenStatesIdx;
    if ((this->param.mapping.Get(base::ATTN_DP).IsEnabled())
            && this->param.isLastLayer) {
        inHiddenStatesIdx = atb_speed::common::GetTensorIdx(this->tensorMap, "in_final_hidden_state");
    } else {
        inHiddenStatesIdx = atb_speed::common::GetTensorIdx(this->tensorMap, "in_hidden_states");
    }

    this->graph.inferShapeFunc = [inHiddenStatesIdx](
        const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(inHiddenStatesIdx);
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(this->graph, operation));

    ATB_SPEED_LOG_DEBUG("Layer Build Graph Success");
    return atb::NO_ERROR;
}

template <typename NormType>
void MoeDecoderLayer<NormType>::SetFusionAttentionLinearParam(atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam)
{
    // QKV param
    fusionAttentionParam.isGroupedQueryAttention = \
        this->param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.isBF16 = this->param.isBF16;
    fusionAttentionParam.isAntiOutlier = this->param.isAntiOutlier.at(0);
    fusionAttentionParam.layerLinearDescs = this->param.linearDescs;
    fusionAttentionParam.layerLinearQuantType = this->param.linearQuantType;
    fusionAttentionParam.layerLinearTransposeType = this->param.linearTransposeType;
    fusionAttentionParam.packQuantType = this->param.packQuantType.at(0);
    fusionAttentionParam.quantGroupSize = this->param.quantGroupSize;
    fusionAttentionParam.matmulBackend = this->param.matmulBackend;
    fusionAttentionParam.supportLora = this->param.enableLora;
    fusionAttentionParam.enablePreFetchWeight = this->param.enablePreFetchWeight;
    fusionAttentionParam.enableMC2 = this->param.enableMC2;
    fusionAttentionParam.loraEnableGMM = this->param.loraEnableGMM;
    fusionAttentionParam.qkvHasBias = this->param.linearHasBias.at(base::QKV_HASBIAS);
    // dense
    fusionAttentionParam.selfAttnHasBias = this->param.linearHasBias.at(base::SELFATTENTION_HASBIAS);
    fusionAttentionParam.supportLcoc = this->param.enableLcoc;
    if (this->param.attnAllreduce) {
        atb_speed::common::ParallelInfo parallelInfo = this->param.mapping.Get(base::ATTN_TP);
        fusionAttentionParam.selfOutLinearTensorParallelInfo.rank = parallelInfo.rank;
        fusionAttentionParam.selfOutLinearTensorParallelInfo.worldSize = parallelInfo.rankIds.size();
        fusionAttentionParam.selfOutLinearTensorParallelInfo.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(
            fusionAttentionParam.selfOutLinearTensorParallelInfo.hcommInfo,
            fusionAttentionParam.selfOutLinearTensorParallelInfo.commDomain);
    }
    if (this->param.enableReduceQuant) {
        fusionAttentionParam.selfOutLinearTensorParallelInfo.quantType = \
            atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL;
        fusionAttentionParam.selfOutLinearTensorParallelInfo.outDataType = ACL_FLOAT16;
    }
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddAttnOutPadding()
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
    gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_attn_out",
                                                                                   "in_attn_padding_idx"});
    gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        this->tensorMap, {"intermediate_attn_out_padding"});
    this->graph.nodes.push_back(gatherNode);
    ATB_SPEED_LOG_DEBUG("Attn out padding success");
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddResidualPadding()
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
    gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"in_hidden_states",
                                                                                   "in_attn_padding_idx"});
    gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        this->tensorMap, {"intermediate_hidden_states_padding"});
    this->graph.nodes.push_back(gatherNode);
    ATB_SPEED_LOG_DEBUG("Residual after attn padding success");
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddResidualSliceNode()
{
    atb::infer::SliceParam sliceParam;
    atb::Node sliceNode;

    sliceParam.offsets.resize(3); // 3: Slice offset dim
    sliceParam.offsets[0] = this->param.mapping.Get(base::ATTN_TP).rank;
    sliceParam.offsets[1] = 0;
    sliceParam.offsets[2] = 0; // 2: dim:2

    sliceParam.size.resize(3); // 3: Slice Size dim
    sliceParam.size[0] = 1;
    sliceParam.size[1] = -1;
    sliceParam.size[2] = -1; // 2: dim:2

    CreateOperation(sliceParam, &sliceNode.operation);
    sliceNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_hidden_states_padding"});
    sliceNode.outTensorIds = \
        atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_hidden_states_slice"});
    
    int64_t attnTPSize = this->param.mapping.Get(base::ATTN_TP).rankIds.size();
    sliceNode.inTensorReshapeFuncs.resize(sliceNode.inTensorIds.size());
    sliceNode.inTensorReshapeFuncs[0] = [attnTPSize] (const atb::Dims &oldShape, atb::Dims &newShape) {
        if (oldShape.dimNum == 2) { // 2: dimNum
            newShape.dimNum = 3; // 3: dimNum
            newShape.dims[0] = attnTPSize;
            newShape.dims[1] = oldShape.dims[0] / attnTPSize;
            newShape.dims[2] = oldShape.dims[1]; // 2: dim 2
        } else {
            newShape.dimNum = 3; // 3: dimNum
            newShape.dims[0] = attnTPSize;
            newShape.dims[1] = oldShape.dims[0] * oldShape.dims[1] / attnTPSize;
            newShape.dims[2] = oldShape.dims[2]; // 2: dim 2
        }
    };
    this->graph.nodes.push_back(sliceNode);
    ATB_SPEED_LOG_DEBUG("Residual after attn slice success");
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddAttnReduceScatter()
{
    atb::Node rsNode;
    atb::infer::ReduceScatterParam rsParam;
    atb_speed::common::ParallelInfo parallelInfo = this->param.mapping.Get(base::ATTN_TP);
    rsParam.rank = parallelInfo.rank;
    rsParam.rankSize = parallelInfo.rankIds.size();
    rsParam.backend = parallelInfo.defaultBackend;
    parallelInfo.InitCommDomain(rsParam.hcclComm, rsParam.commDomain);
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(rsParam, &rsNode.operation));
    rsNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_attn_out_padding"});
    rsNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_attn_out_scatter"});
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(this->graph));
    this->graph.nodes.push_back(rsNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(this->graph));
    ATB_SPEED_LOG_DEBUG("Reduce scatter after attn success");
    return atb::NO_ERROR;
}

template<typename NormType>
void MoeDecoderLayer<NormType>::SetSelfNormParam(atb_speed::common::NormLinearParam<NormType> &selfNormParam)
{
    atb_speed::common::MlpParam<NormType> mlpParam;
    atb_speed::base::DecoderLayer<NormType>::SetMlpParam(mlpParam);
    selfNormParam.isAntiOutlier = atb_speed::common::CheckAntiOutlier(mlpParam.packQuantType);
    selfNormParam.normHasBias = this->param.normHasBias;
    selfNormParam.enableAddNorm = mlpParam.enableAddNorm;
    selfNormParam.normParamType = mlpParam.normParamType;
    selfNormParam.normQuantParamType = mlpParam.normQuantParamType;
}


template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::SetAddNormNode()
{
    atb::Node selfAddNormNode;
    selfAddNormNode.operation = new atb_speed::common::AddRmsNormOperation("AddRmsNormNode", 1e-6);
    std::string inHiddenStatesStr = "in_hidden_states";
    std::string intermediateAttnOutStr = "intermediate_attn_out";
    selfAddNormNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(this->tensorMap, {inHiddenStatesStr,
                                                        intermediateAttnOutStr,
                                                        "in_post_attn_norm_weight"});
    selfAddNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {
        "intermediate_norm_out",
        "intermediate_norm_rstd_out",
        inHiddenStatesStr});

    this->graph.nodes.push_back(selfAddNormNode);
    ATB_SPEED_LOG_DEBUG("create SetAttentionResidualAddNormNode");
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddFusionAttentionResidualAdd()
{
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Node selfResidualAddNode;
    std::string inHiddenStatesStr = "in_hidden_states";
    std::string intermediateAttnOutStr = "intermediate_attn_out";
    if (this->param.hasAttnComm) {
        inHiddenStatesStr = this->param.attnReduceScatter ?
            "intermediate_hidden_states_slice" : "intermediate_hidden_states_padding";
        intermediateAttnOutStr = this->param.attnReduceScatter ?
            "intermediate_attn_out_scatter" : "intermediate_attn_out_padding";
    }
    
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &selfResidualAddNode.operation));
    selfResidualAddNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(this->tensorMap, {inHiddenStatesStr, intermediateAttnOutStr});
    if (this->param.attnReduceScatter) {
        selfResidualAddNode.inTensorReshapeFuncs.resize(selfResidualAddNode.inTensorIds.size());
        selfResidualAddNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 2; // 2: dimNum
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
            newShape.dims[1] = oldShape.dims[2]; // 2: dim 2
        };
    }
    selfResidualAddNode.outTensorIds = \
        atb_speed::common::GetTensorIdxList(this->tensorMap, {inHiddenStatesStr});
    this->graph.nodes.push_back(selfResidualAddNode);
    
    return atb::NO_ERROR;
}

template<typename NormType>
std::map<std::string, uint32_t> MoeDecoderLayer<NormType>::ConstructNormTensorMap() const
{
    std::vector<std::string> targetNames = {
        "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_bias", "in_scale", "in_offset",
        "intermediate_norm", "out_add", "in_residual_input"
    };
    std::vector<std::string> originNames = {};
    if (this->param.hasAttnComm) {
        std::string inHiddenStatesStr = this->param.attnReduceScatter ?
            "intermediate_hidden_states_slice" : "intermediate_hidden_states_padding";
        std::string intermediateAttnOutStr = this->param.attnReduceScatter ?
            "intermediate_attn_out_scatter" : "intermediate_attn_out_padding";
        std::string intermediateNormOutStr = this->param.attnAllGather ?
            "intermediate_norm_out_partial" : "intermediate_norm_out_with_padding";
        originNames = {inHiddenStatesStr, "in_post_attn_norm_weight",
            "in_post_attn_norm_bias", "in_post_attn_norm_new_weight",
            "in_post_attn_norm_new_bias", "in_mlp_gateup_scale", "in_mlp_gateup_offset",
            intermediateNormOutStr, inHiddenStatesStr, intermediateAttnOutStr};
    } else {
        originNames = {"in_hidden_states", "in_post_attn_norm_weight", "in_post_attn_norm_bias",
            "in_post_attn_norm_new_weight", "in_post_attn_norm_new_bias", "in_mlp_gateup_scale",
            "in_mlp_gateup_offset", "intermediate_norm_out", "in_hidden_states", "intermediate_attn_out"};
    }
    std::map<std::string, uint32_t> normTensorMap;
    for (size_t i = 0; i < targetNames.size(); ++i) {
        normTensorMap[targetNames.at(i)] = atb_speed::common::GetTensorIdx(this->tensorMap, originNames.at(i));
    }
    return normTensorMap;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddSelfNorm()
{
    atb_speed::common::NormLinearParam<NormType> selfNormParam;
    SetSelfNormParam(selfNormParam);
    std::map<std::string, uint32_t> selfNormTensorMap = ConstructNormTensorMap();
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::InsertNorm(this->graph, selfNormParam, selfNormTensorMap));
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddAttnAllGather()
{
    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    if (this->param.attnReduceScatter) {
        atb_speed::common::ParallelInfo parallelInfo = this->param.mapping.Get(base::MLP_TP);
        allGatherParam.rank = parallelInfo.rank;
        allGatherParam.rankSize = parallelInfo.rankIds.size();
        allGatherParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    } else {
        atb_speed::common::ParallelInfo parallelInfo = this->param.mapping.Get(base::ATTN_DP);
        allGatherParam.rank = parallelInfo.rank;
        allGatherParam.rankSize = parallelInfo.rankIds.size();
        allGatherParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    }

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
    allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        this->tensorMap, {"intermediate_norm_out_partial"});
    allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        this->tensorMap, {"intermediate_norm_out_with_padding"});
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(this->graph));
    this->graph.nodes.push_back(allGatherNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(this->graph));

    ATB_SPEED_LOG_DEBUG("Attn AllGather calculation success");
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddAttnOutUnpadding()
{
    atb::Node unpadNode;
    atb::infer::GatherParam unpadParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(unpadParam, &unpadNode.operation));
    unpadNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        this->tensorMap, {"intermediate_norm_out_with_padding", "in_attn_unpadding_idx"});
    unpadNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_norm_out"});
    unpadNode.inTensorReshapeFuncs.reserve(unpadNode.inTensorIds.size());
    unpadNode.inTensorReshapeFuncs.resize(unpadNode.inTensorIds.size());
    unpadNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2:New shape's dimnum is 2
        if (oldShape.dimNum == 3) { // 3:Old shape's dimnum is 3
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 0, 0, 1: Dim id
            newShape.dims[1] = oldShape.dims[2]; // 1, 2: Dim id
        } else {
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = oldShape.dims[1]; // 1, 2: Dim id
        }
    };
    this->graph.nodes.push_back(unpadNode);
    ATB_SPEED_LOG_DEBUG("Attn unpadding calculation success");
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddPostAttentionProcess()
{
    if (this->param.hasAttnComm) {
        CHECK_OPERATION_STATUS_RETURN(this->AddAttnOutPadding());
        CHECK_OPERATION_STATUS_RETURN(this->AddResidualPadding());
        CHECK_OPERATION_STATUS_RETURN(this->AddResidualSliceNode());
        if (this->param.attnReduceScatter) {
            CHECK_OPERATION_STATUS_RETURN(this->AddAttnReduceScatter());
        }
        CHECK_OPERATION_STATUS_RETURN(this->AddFusionAttentionResidualAdd());
        CHECK_OPERATION_STATUS_RETURN(this->AddSelfNorm());
        if (this->param.attnAllGather) {
            CHECK_OPERATION_STATUS_RETURN(this->AddAttnAllGather());
        }
        CHECK_OPERATION_STATUS_RETURN(this->AddAttnOutUnpadding());
    } else {

        if (this->param.enableIntraLayerAddNorm) {
            // node: residual + norm
            CHECK_OPERATION_STATUS_RETURN(this->SetAddNormNode());
        } else {
            ATB_SPEED_LOG_DEBUG("moelayer_disableIntraLayerAddNorm ");
            CHECK_OPERATION_STATUS_RETURN(this->AddFusionAttentionResidualAdd());
            CHECK_OPERATION_STATUS_RETURN(this->AddSelfNorm());
        }
    }
    return atb::NO_ERROR;
}

template <typename NormType>
void MoeDecoderLayer<NormType>::SetSharedExpertParam(atb_speed::common::SharedExpertParam &sharedExpertParam)
{
    sharedExpertParam.transposeGateup = this->param.transpose;
    sharedExpertParam.transposeDown = this->param.transpose;
    sharedExpertParam.hasSharedExpertGate = false;
    sharedExpertParam.mlpLinearQuantType = this->param.mlpLinearQuantType;
    sharedExpertParam.mlpLinearTransposeType = this->param.mlpLinearTransposeType;
    sharedExpertParam.packQuantType = this->param.packQuantType.at(1);
    sharedExpertParam.isBF16 = this->param.isBF16;
    sharedExpertParam.supportSwiGLU = this->param.enableSwiGLU;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddMlpExpert(const atb_speed::common::SharedExpertParam &mlpExpertParam)
{
    atb::Node mlpExpertNode;
    CHECK_OPERATION_STATUS_RETURN(
        atb_speed::common::CreateSharedExpertOperation(mlpExpertParam, &mlpExpertNode.operation));
    mlpExpertNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {
        "intermediate_norm_out", "in_mlp_gateup_weight_shared_expert", "in_mlp_gateup_bias_shared_expert",
        "in_mlp_gateup_descale_shared_expert", "in_mlp_gateup_offset_shared_expert",
        "in_mlp_gateup_scale_shared_expert", "in_mlp_gateup_compress_idx_shared_expert",
        "in_mlp_down_weight_shared_expert", "in_mlp_down_bias_shared_expert", "in_mlp_down_descale_shared_expert",
        "in_mlp_down_offset_shared_expert", "in_mlp_down_scale_shared_expert", "in_mlp_down_compress_idx_shared_expert",
        "in_shared_expert_gate_weight", "in_shared_expert_gate_bias", "in_shared_expert_gate_descale",
        "in_shared_expert_gate_offset", "in_shared_expert_gate_scale", "in_shared_expert_gate_compress_idx"
    });
    if (this->param.isDenseLayer) {
        mlpExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
    } else {
        mlpExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"shared_expert_out"});
    }
    this->graph.nodes.push_back(mlpExpertNode);
    ATB_SPEED_LOG_DEBUG("mlp expert calculation success");
    return atb::NO_ERROR;
}

template <typename NormType>
void MoeDecoderLayer<NormType>::SetSparseMoeParam(atb_speed::common::SparseMoeParam &sparseMoeParam)
{
    sparseMoeParam.transpose = this->param.transpose;
    sparseMoeParam.numOfExperts = this->param.numOfExperts;
    sparseMoeParam.num = this->param.numOfSelectedExperts;
    sparseMoeParam.expertParallelDegree = this->param.expertParallelDegree;
    sparseMoeParam.processLogits = this->param.processLogits;
    sparseMoeParam.supportSwiGLU = this->param.enableSwiGLU;
    sparseMoeParam.routingMethod = this->param.routingMethod;
    sparseMoeParam.moeLinearQuantType = this->param.moeLinearQuantType;
    sparseMoeParam.isBF16 = this->param.isBF16;
    sparseMoeParam.enableFusedRouting = this->param.enableFusedRouting;
    sparseMoeParam.hasMoeEp = this->param.hasMoeEp;
    sparseMoeParam.gateUpTransposeB = this->param.moeLinearTransposeType[moeGateupLinearIndex];
    sparseMoeParam.downTransposeB = this->param.moeLinearTransposeType[moeDownLinearIndex];

    sparseMoeParam.numOfDeviceExperts = this->param.numOfDeviceExperts;
    sparseMoeParam.numOfGroups = this->param.numOfGroups;
    sparseMoeParam.topkGroups = this->param.topkGroups;
    sparseMoeParam.scaledTopk = this->param.scaledTopk;
    sparseMoeParam.enableInitRoutingCutoff = this->param.enableInitRoutingCutoff;
    sparseMoeParam.isDynamicEp = this->param.isDynamicEp;
    sparseMoeParam.deviceExpert = this->param.deviceExpert;
    sparseMoeParam.routedScalingFactor = this->param.routedScalingFactor;
    if (this->param.moePackQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED) {
        sparseMoeParam.packQuantType = this->param.packQuantType.at(1);
    } else {
        sparseMoeParam.packQuantType = this->param.moePackQuantType;
    }
    sparseMoeParam.quantGroupSize = this->param.quantGroupSize;
    sparseMoeParam.enableInitQuant = this->param.enableInitQuant;
    sparseMoeParam.enableInitRoutingV3 = this->param.enableInitRoutingV3;
    sparseMoeParam.enableSwigluQuant = this->param.enableSwigluQuant;
    sparseMoeParam.swigluBackend = this->param.swigluBackend;
    sparseMoeParam.enableFusedTopk = this->param.enableFusedTopk;
    sparseMoeParam.enableExpertCumSumOutput = this->param.enableExpertCumSumOutput;
    sparseMoeParam.enableATBGateMatmul = this->param.enableATBGateMatmul;
    sparseMoeParam.enableGMMSwigluQuant = this->param.enableGMMSwigluQuant;
    sparseMoeParam.enableAtlasGMMFused = this->param.enableAtlasGMMFused;
    sparseMoeParam.enableCVOverlap = this->param.enableCVOverlap;
    sparseMoeParam.enableLoadBalance = this->param.enableLoadBalance;
    sparseMoeParam.enableEPWB = this->param.enableEPWB;
    sparseMoeParam.numOfRedundantExpert = this->param.numOfRedundantExpert;
    sparseMoeParam.numDanglingSharedExperts = this->param.numDanglingSharedExperts;
    sparseMoeParam.maxDecodeDpTokenSize = this->param.maxDecodeDpTokenSize;
    sparseMoeParam.enableMoeDistribute = !this->param.isPrefill && this->param.enableAllToAllMC2 && this->param.isDynamicEp;
    sparseMoeParam.enableGatingDp = this->param.enableGatingDp && this->param.isPrefill;  // h3p gatingdp for moe
    sparseMoeParam.enableGatingShift = this->param.enableGatingDp && !this->param.isPrefill;  // h3p gatingshift for decode
    sparseMoeParam.enableGatingOverlap = sparseMoeParam.enableGatingDp &&
                                        this->param.enableSharedExpertOverlap;  // h3p Gating overlap
}

template <typename NormType>
void MoeDecoderLayer<NormType>::SetSparseMoeCommParam(atb_speed::common::SparseMoeParam &sparseMoeParam)
{
    sparseMoeParam.backend = this->param.mapping.Get(base::MOE_EP).defaultBackend;
    sparseMoeParam.hasMoeEp = this->param.mapping.Get(base::MOE_EP).IsEnabled();
    sparseMoeParam.moeEpRank = this->param.mapping.Get(base::MOE_EP).rank;
    sparseMoeParam.moeEpSize = this->param.mapping.Get(base::MOE_EP).rankIds.size();
    sparseMoeParam.moeEpRankTableFile = "";
    this->param.mapping.Get(base::MOE_EP).InitCommDomain(sparseMoeParam.hcclComm, sparseMoeParam.moeEpDomain);
    
    sparseMoeParam.mlpTpBackend = this->param.mapping.Get(base::MLP_TP).defaultBackend;
    sparseMoeParam.hasMlpTp = this->param.mapping.Get(base::MLP_TP).IsEnabled();
    sparseMoeParam.mlpTpRank = this->param.mapping.Get(base::MLP_TP).rank;
    sparseMoeParam.mlpTpSize = this->param.mapping.Get(base::MLP_TP).rankIds.size();
    sparseMoeParam.mlpTpRankTableFile = "";
    this->param.mapping.Get(base::MLP_TP).InitCommDomain(sparseMoeParam.hcclTpComm, sparseMoeParam.mlpTpDomain);
    
    sparseMoeParam.enableLcocAll2All = this->param.enableLcocAll2All;
    sparseMoeParam.enableDispatchCombineV2 = this->param.enableDispatchCombineV2;
    if (sparseMoeParam.enableMoeDistribute) {
        sparseMoeParam.moeEpDomain = this->param.dispatchAndCombinecommDomain;
        sparseMoeParam.hcclComm = this->param.dispatchAndCombineHcclComm;
    }
    if (sparseMoeParam.enableLcocAll2All) {
        this->param.mapping.Get(base::MOE_EP).InitCommDomain(
            sparseMoeParam.lcclMoeEpHcclComm, sparseMoeParam.lcclMoeEpDomain, "lccl");
    }
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddMoe()
{
    atb::Node moeNode;
    atb_speed::common::SparseMoeParam sparseMoeParam;
    this->SetSparseMoeParam(sparseMoeParam);
    this->SetSparseMoeCommParam(sparseMoeParam);
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateSparseMoeOperation(sparseMoeParam, &moeNode.operation));
    std::vector<std::string> sparseMoeInTensorNames = {
        "intermediate_norm_out", "block_sparse_moe_gate_weight", "block_sparse_moe_gate_bias", "block_sparse_moe_gate_descale",
        "block_sparse_moe_gate_offset", "block_sparse_moe_gate_scale", "block_sparse_moe_gate_compress_idx",
        "in_mlp_gateup_weight", "in_mlp_gateup_bias", "in_mlp_gateup_descale", "in_mlp_gateup_offset",
        "in_mlp_gateup_scale", "in_mlp_gateup_compress_idx", "in_mlp_down_weight", "in_mlp_down_bias",
        "in_mlp_down_descale", "in_mlp_down_offset", "in_mlp_down_scale", "in_mlp_down_compress_idx", "expert_array",
        "expert_group", "one_hot", "zero_hot"
    };
    if (this->param.mapping.Get(base::MOE_EP).IsEnabled()) {
        sparseMoeInTensorNames.push_back("in_start_expert_idx");
        sparseMoeInTensorNames.push_back("in_device_expert_count");
        sparseMoeInTensorNames.push_back("in_ffn_padding_idx");
        if (this->param.isDynamicEp) {
            sparseMoeInTensorNames.push_back("in_lty_idx");
            sparseMoeInTensorNames.push_back("in_moe_idx");
        }
    }
    moeNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, sparseMoeInTensorNames);
    moeNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {
        this->param.hasSharedExpert ? "intermediate_moe_out" : "intermediate_mlp_out"});
    
    this->graph.nodes.push_back(moeNode);
    ATB_SPEED_LOG_DEBUG("Create Moe success");

    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddExpertAdd()
{
    atb::Node expertAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &expertAddNode.operation));
    expertAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_moe_out", "shared_expert_out"});
    expertAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
    this->graph.nodes.push_back(expertAddNode);
    ATB_SPEED_LOG_DEBUG("create add operation");
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddFFN()
{
    if (this->param.isDenseLayer) {
        atb_speed::common::SharedExpertParam mlpExpertParam;
        SetSharedExpertParam(mlpExpertParam);
        CHECK_OPERATION_STATUS_RETURN(this->AddMlpExpert(mlpExpertParam));
    } else {
        if (this->param.hasSharedExpert) {
            atb_speed::common::SharedExpertParam sharedExpertParam;
            SetSharedExpertParam(sharedExpertParam);
            sharedExpertParam.hasSharedExpertGate = this->param.hasSharedExpertGate;
            CHECK_OPERATION_STATUS_RETURN(this->AddMlpExpert(sharedExpertParam));
        }
        CHECK_OPERATION_STATUS_RETURN(this->AddMoe());
        if (this->param.hasSharedExpert) {
            CHECK_OPERATION_STATUS_RETURN(this->AddExpertAdd());
        }
    }
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddFFNOutPadding()
{
    atb::Node padNode;
    atb::infer::GatherParam padParam;
    atb::CreateOperation(padParam, &padNode.operation);
    padNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {
        this->param.hasAttnComm ? "intermediate_mlp_out" : this->param.ffnReduceScatter ?
            "intermediate_mlp_out_scatter" : "intermediate_mlp_out",
        this->param.hasAttnComm ? "in_ffn_padding_idx" : "in_attn_padding_idx"});
    padNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {
        this->param.ffnAllGather ? "intermediate_mlp_out_padding_partial" : "intermediate_mlp_out_padding"});
    this->graph.nodes.push_back(padNode);
    ATB_SPEED_LOG_DEBUG("Create ffn pad Node");
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddMoeAllReduce()
{
    atb::Node moeAllReduceNode;
    atb::infer::AllReduceParam allReduceParam;
    if (this->param.isDenseLayer) {
        atb_speed::common::ParallelInfo parallelInfo = this->param.mapping.Get(base::ATTN_TP);
        allReduceParam.rank = parallelInfo.rank;
        allReduceParam.rankSize = parallelInfo.rankIds.size();
        allReduceParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allReduceParam.hcclComm, allReduceParam.commDomain);
    } else {
        atb_speed::common::ParallelInfo parallelInfo = this->param.mapping.Get(base::MLP_TP);
        allReduceParam.rank = parallelInfo.rank;
        allReduceParam.rankSize = parallelInfo.rankIds.size();
        allReduceParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allReduceParam.hcclComm, allReduceParam.commDomain);
    }
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(allReduceParam, &moeAllReduceNode.operation));
    moeAllReduceNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
    moeAllReduceNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
    this->graph.nodes.push_back(moeAllReduceNode);
    ATB_SPEED_LOG_DEBUG("Create FFN allreduce");

    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddFFNReduceScatter()
{
    atb::Node reduceScatterNode;
    atb::infer::ReduceScatterParam reduceScatterParam;
    atb_speed::common::ParallelInfo parallelInfo = this->param.mapping.Get(base::MLP_TP);
    reduceScatterParam.rank = parallelInfo.rank;
    reduceScatterParam.rankSize = parallelInfo.rankIds.size();
    reduceScatterParam.backend = parallelInfo.defaultBackend;
    parallelInfo.InitCommDomain(reduceScatterParam.hcclComm, reduceScatterParam.commDomain);

    CreateOperation(reduceScatterParam, &reduceScatterNode.operation);

    reduceScatterNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {this->param.ffnAllGather ?
        "intermediate_mlp_out_padding_partial" : "intermediate_mlp_out_padding"});
    reduceScatterNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {
        "intermediate_mlp_out_scatter"});
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(this->graph));
    this->graph.nodes.push_back(reduceScatterNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(this->graph));
    ATB_SPEED_LOG_DEBUG("Create attn allreduce");
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddFFNAllGatherNode()
{
    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    if (!this->param.isLastLayer) {
        atb_speed::common::ParallelInfo parallelInfo = this->param.mapping.Get(base::ATTN_TP);
        allGatherParam.rank = parallelInfo.rank;
        allGatherParam.rankSize = parallelInfo.rankIds.size();
        allGatherParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    } else {
        atb_speed::common::ParallelInfo parallelInfo = this->param.mapping.Get(base::MLP_TP);
        allGatherParam.rank = parallelInfo.rank;
        allGatherParam.rankSize = parallelInfo.rankIds.size();
        allGatherParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    }

    CreateOperation(allGatherParam, &allGatherNode.operation);

    allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {
        this->param.ffnReduceScatter ? "intermediate_mlp_out_scatter" : "intermediate_mlp_out_padding_partial"});
    allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        this->tensorMap, {"intermediate_mlp_out_padding"});

    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(this->graph));
    this->graph.nodes.push_back(allGatherNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(this->graph));
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddFFNOutUnPadding()
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
    gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {
        this->param.ffnAllGather ? "intermediate_mlp_out_padding" : "intermediate_mlp_out_scatter",
        this->param.isLastLayer ? "in_lm_head_skip_padding_token_indices" : "in_ffn_unpadding_idx"});
    gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"out"});
    if (this->param.ffnAllGather) {
        gatherNode.inTensorReshapeFuncs.resize(gatherNode.inTensorIds.size());
        gatherNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 2; // 2: dimNum
                newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
                newShape.dims[1] = oldShape.dims[2]; // 2: dim 2
        };
    }
    this->graph.nodes.push_back(gatherNode);
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddMlpResidualAdd()
{
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Node mlpResidualAddNode;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &mlpResidualAddNode.operation));
    std::vector<std::string> mlpResidualAddInTensorNames = {
        !this->param.hasAttnComm ?
            "in_hidden_states" : (this->param.attnReduceScatter ?
            "intermediate_hidden_states_slice" : "intermediate_hidden_states_padding"),
        this->param.ffnReduceScatter ?
            "intermediate_mlp_out_scatter" : (!this->param.ffnAllGather ?
                "intermediate_mlp_out" : this->param.hasAttnComm ?
                "intermediate_mlp_out_padding_partial" : "intermediate_mlp_out")};
    std::vector<std::string> mlpResidualAddOutTensorNames = {
        this->param.hasFfnComm ? mlpResidualAddInTensorNames[1] : "out"};

    mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        this->tensorMap, mlpResidualAddInTensorNames);
    mlpResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        this->tensorMap, mlpResidualAddOutTensorNames);
    this->graph.nodes.push_back(mlpResidualAddNode);
    return atb::NO_ERROR;
}


template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddPostFFNProcess()
{
    if (this->param.ffnAllreduce) {
        CHECK_OPERATION_STATUS_RETURN(this->AddMoeAllReduce());
    }
    if (this->param.hasFfnComm && this->param.hasAttnComm) {
        CHECK_OPERATION_STATUS_RETURN(this->AddFFNOutPadding());
    }
    if (this->param.ffnReduceScatter) {
        CHECK_OPERATION_STATUS_RETURN(this->AddFFNReduceScatter());
    }

    CHECK_OPERATION_STATUS_RETURN(this->AddMlpResidualAdd());
    if (this->param.ffnAllGather) {
        if (!this->param.hasAttnComm) {
            CHECK_OPERATION_STATUS_RETURN(this->AddFFNOutPadding());
        }
        CHECK_OPERATION_STATUS_RETURN(this->AddFFNAllGatherNode());
    }
    if (this->param.hasFfnComm) {
        CHECK_OPERATION_STATUS_RETURN(this->AddFFNOutUnPadding());
    }
    return atb::NO_ERROR;
}

template class MoeDecoderLayer<atb::infer::RmsNormParam>;
template class MoeDecoderLayer<atb::infer::LayerNormParam>;

} // namespace moe
} // namespace atb_speed
