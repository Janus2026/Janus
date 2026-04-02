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
#include "models/moe/layer/decoder_layer.h"
#include "models/glm/layer/moe_decoder_layer.h"
#include "models/deepseekv2/operation/latent_attention.h"

#include "models/base/util.h"

namespace atb_speed {
namespace glm {

template <typename NormType>
MoeDecoderLayer<NormType>::MoeDecoderLayer(
    const MoeLayerParam &param) : atb_speed::moe::MoeDecoderLayer<NormType>(param) {this->param = param;};

template <typename NormType>
void MoeDecoderLayer<NormType>::SetFusionAttentionParam(
    atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam)
{
    atb_speed::moe::MoeDecoderLayer<NormType>::SetFusionAttentionParam(fusionAttentionParam);
    fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::HALF_ROTARY;
    fusionAttentionParam.ropeParam.rotaryCoeff = 2; // 2: rotary coeff
}


std::map<std::string, std::vector<std::string>> LayerAttnInTensorDefaultCandidates()
{
    std::map<std::string, std::vector<std::string>> AttnInTensorCandidates = {
        {"base_attn_weight", {
            "in_qkv_weight_0", "in_qkv_bias_0", "in_qkv_descale_0", 
            "in_qkv_offset_0", "in_qkv_scale_0", "in_qkv_compress_idx_0",
            "in_qkv_weight_1", "in_qkv_bias_1", "in_qkv_descale_1", 
            "in_qkv_offset_1", "in_qkv_scale_1", "in_qkv_compress_idx_1",
            "in_qkv_weight_2", "in_qkv_bias_2", "in_qkv_descale_2", 
            "in_qkv_offset_2", "in_qkv_scale_2", "in_qkv_compress_idx_2",

            // shape: [hiddenSize, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
            "in_qkv_dense_weight", "in_qkv_dense_bias", "in_qkv_dense_descale", "in_qkv_dense_offset",
            "in_qkv_dense_scale", "in_qkv_dense_compress_idx"}},
        {"default", {
            "in_hidden_states",  // shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
            "in_cos_embedding", "in_sin_embedding", "in_attention_mask", "in_k_cache", "in_k_rope_cache", "in_seq_len",
            "in_token_offset", "in_layer_id", "in_block_tables", "in_slots", 
            "in_q_len"  //ADD
            }},
        {"attn_weight", {
            "in_q_proj_a_weight", "in_q_proj_a_bias", "in_q_proj_a_descale", 
            "in_q_proj_a_offset", "in_q_proj_a_scale", "in_q_proj_a_compress_idx", 

            "in_q_proj_b_weight", "in_q_proj_b_bias", "in_q_proj_b_descale", 
            "in_q_proj_b_offset", "in_q_proj_b_scale", "in_q_proj_b_compress_idx", 
            
            "in_kv_proj_with_mqa_weight", "in_kv_proj_with_mqa_bias", "in_kv_proj_with_mqa_descale", 
            "in_kv_proj_with_mqa_offset", "in_kv_proj_with_mqa_scale", "in_kv_proj_with_mqa_compress_idx", 

            "in_k_proj_b_weight", "in_k_proj_b_bias", "in_k_proj_b_descale",
            "in_k_proj_b_offset", "in_k_proj_b_scale", "in_k_proj_b_compress_idx",

            "in_v_proj_b_weight", "in_v_proj_b_bias", "in_v_proj_b_descale",
            "in_v_proj_b_offset", "in_v_proj_b_scale", "in_v_proj_b_compress_idx",

            "in_attention_out_weight", "in_attention_out_bias", "in_attention_out_descale", "in_attention_out_offset", "in_attention_out_scale", "in_attention_out_compress_idx"}
        },
        {"in_attn_weight", {                //attentionNode input list
            "in_q_proj_a_weight", "in_q_proj_a_bias", "in_q_proj_a_descale", 
            "in_q_proj_a_offset", "in_q_proj_a_scale", "in_q_proj_a_compress_idx", 

            "in_q_proj_a_layernorm_weight", "in_q_proj_a_layernorm_bias",

            "in_q_proj_b_weight", "in_q_proj_b_bias", "in_q_proj_b_descale", 
            "in_q_proj_b_offset", "in_q_proj_b_scale", "in_q_proj_b_compress_idx", 
            
            "in_kv_proj_with_mqa_weight", "in_kv_proj_with_mqa_bias", "in_kv_proj_with_mqa_descale", 
            "in_kv_proj_with_mqa_offset", "in_kv_proj_with_mqa_scale", "in_kv_proj_with_mqa_compress_idx", 
            
            "in_kv_proj_a_layernorm_weight", "in_kv_proj_a_layernorm_bias",

            "in_k_proj_b_weight", "in_k_proj_b_bias", "in_k_proj_b_descale",
            "in_k_proj_b_offset", "in_k_proj_b_scale", "in_k_proj_b_compress_idx",

            "in_v_proj_b_weight", "in_v_proj_b_bias", "in_v_proj_b_descale",
            "in_v_proj_b_offset", "in_v_proj_b_scale", "in_v_proj_b_compress_idx",

            "in_attention_out_weight", "in_attention_out_bias", "in_attention_out_descale", "in_attention_out_offset",
            "in_attention_out_scale", "in_attention_out_compress_idx"}
        },
        {"qk_norm", {
            "in_q_proj_a_layernorm_weight", "in_q_proj_a_layernorm_bias",
            "in_kv_proj_a_layernorm_weight", "in_kv_proj_a_layernorm_bias"}},
        {"indexer_weight",{
            "in_indexer_proj_wq_b_weight", "in_indexer_proj_wq_b_bias", "in_indexer_proj_wq_b_descale",
            "in_indexer_proj_wq_b_offset", "in_indexer_proj_wq_b_scale","in_indexer_proj_wq_b_compress_idx",
            "in_indexer_proj_wk_weight", "in_indexer_proj_wk_bias", "in_indexer_proj_wk_descale",
            "in_indexer_proj_wk_offset", "in_indexer_proj_wk_scale","in_indexer_proj_wk_compress_idx",
            "in_indexer_proj_k_norm_weight", "in_indexer_proj_k_norm_bias",
            "in_indexer_proj_weight", "in_indexer_proj_bias", "in_indexer_proj_descale",
            "in_indexer_proj_offset", "in_indexer_proj_scale","in_indexer_proj_compress_idx",
            "in_q_proj_a_recompte_weight", "in_q_proj_a_recompte_bias", "in_q_proj_a_recompte_descale",
            "in_q_proj_a_recompte_offset", "in_q_proj_a_recompte_scale","in_q_proj_a_recompte_compress_idx"
            }},
        {"prefixcache", {
            "in_history_compressed_kv", "in_history_k_rope", "ring_cur_seqlen", "ring_cache_seqlen"}},
    };
    return AttnInTensorCandidates;
}

template <typename NormType>
void MoeDecoderLayer<NormType>::ConstructInTensorMap()
{
    this->inTensorList.clear();
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "input_norm_weight", this->inTensorList);
    if (!this->param.useMLA) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "attn_weight", this->inTensorList);
    }
    else {
        atb_speed::common::AddTensorToList(LayerAttnInTensorDefaultCandidates(), "attn_weight", this->inTensorList);
    }

    atb_speed::common::AddTensorToList(this->inTensorCandidates, "post_attn_norm_weight", this->inTensorList);
    if (this->param.hasSharedExpert || this->param.isDenseLayer) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "shared_expert_weight", this->inTensorList);
    }
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "moe_weight", this->inTensorList);
    if (this->param.useQKNorm) {
        if (!this->param.useMLA) {
            atb_speed::common::AddTensorToList(this->inTensorCandidates, "qk_norm", this->inTensorList);
        }
        else {
            atb_speed::common::AddTensorToList(LayerAttnInTensorDefaultCandidates(), "qk_norm", this->inTensorList);
        }
    }
    if (!this->param.useMLA) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "default", this->inTensorList);
    }
    else {
        atb_speed::common::AddTensorToList(LayerAttnInTensorDefaultCandidates(), "default", this->inTensorList);
    }
    if (this->param.hasAttnDp) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "attn_dp", this->inTensorList);
    }
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "default_moe", this->inTensorList);

    if (!this->param.useMLA) {
        if (this->param.enableSpeculate || this->param.enableSplitFuse || this->param.enablePrefixCache) {
            atb_speed::common::AddTensorToList(this->inTensorCandidates, "q_len", this->inTensorList);
        }
    }
    else {
        if (this->param.enablePrefixCache) {
            atb_speed::common::AddTensorToList(LayerAttnInTensorDefaultCandidates(), "prefixcache", this->inTensorList);
        }
    }    

    // if (this->param.enableSpeculate || this->param.enableSplitFuse || this->param.enablePrefixCache) {
    //     atb_speed::common::AddTensorToList(this->inTensorCandidates, "q_len", this->inTensorList);
    // }
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "parallel", this->inTensorList);
    if (this->param.enableAclGraphPagedAttention && !this->param.isPrefill) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "acl_graph", this->inTensorList);
    }
}


void SetRmsNormParam(
    atb_speed::deepseekV2::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
    const glm::MoeLayerParam &param)
{
    atb::infer::RmsNormParam attenRmsNormParam;
    attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormParam.normParam.epsilon = param.normEps;
    latentAttentionParam.normParamType = attenRmsNormParam;
}

void SetRmsNormQuantParam(
    atb_speed::deepseekV2::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
    const glm::MoeLayerParam &param)
{
    atb::infer::RmsNormParam attenRmsNormQuantParam;
    attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormQuantParam.normParam.epsilon = param.normEps;
    attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    latentAttentionParam.normQuantParamType = attenRmsNormQuantParam;
}

void SetAttnCpParam(
    atb_speed::deepseekV2::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
    const glm::MoeLayerParam &param)
{
    if (param.mapping.Get(base::ATTN_CP).IsEnabled()) {
        latentAttentionParam.contextParallelInfo = param.mapping.Get(base::ATTN_CP);
        latentAttentionParam.ringMLAParam.headNum = param.numAttentionHeadsPerRank;
        latentAttentionParam.ringMLAParam.kvHeadNum = param.numAttentionHeadsPerRank;
        latentAttentionParam.ringMLAParam.qkScale = param.softmaxScale;
    }
}

void SetAttnInnerSpParam(
    atb_speed::deepseekV2::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
    const glm::MoeLayerParam &param)
{
    if (param.mapping.Get(base::ATTN_INNER_SP).IsEnabled()) {
        latentAttentionParam.hasAttnInnerSp = param.mapping.Get(base::ATTN_INNER_SP).IsEnabled();
        latentAttentionParam.attnSpRank = param.mapping.Get(base::ATTN_INNER_SP).rank;
        latentAttentionParam.attnSpSize = param.mapping.Get(base::ATTN_INNER_SP).rankIds.size();
        latentAttentionParam.attnSpRankTableFile = "";
        latentAttentionParam.attnSpBackend = "lccl";
        param.mapping.Get(base::ATTN_INNER_SP).InitCommDomain(
            latentAttentionParam.attnSpHcclComm,
            latentAttentionParam.attnSpDomain,
            "lccl");

        latentAttentionParam.pageAttentionParam.headNum = \
            param.numAttentionHeadsPerRank * param.mapping.Get(base::ATTN_INNER_SP).rankIds.size();
    }
}

atb::Status SetLatentAttentionInnerCommParam(
    atb_speed::deepseekV2::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
    const glm::MoeLayerParam &param)
{
    latentAttentionParam.enableExtraOprojTp = param.enableExtraOprojTp;
    latentAttentionParam.selfOutLinearInnerTensorParallelInfo = param.mapping.Get(base::ATTN_O_PROJ_TP);
    latentAttentionParam.attnOprojPrefetch = param.attnOprojPrefetch;
    latentAttentionParam.enableFusedMLA = param.enableFusedMLA;
    latentAttentionParam.actual_headNum = param.actual_headNum;
    if (param.attnAllreduce) {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
        latentAttentionParam.selfOutLinearTensorParallelInfo.rank = parallelInfo.rank;
        latentAttentionParam.selfOutLinearTensorParallelInfo.worldSize = parallelInfo.rankIds.size();
        latentAttentionParam.selfOutLinearTensorParallelInfo.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(
            latentAttentionParam.selfOutLinearTensorParallelInfo.hcommInfo,
            latentAttentionParam.selfOutLinearTensorParallelInfo.commDomain);
    }
    return atb::NO_ERROR;
}

atb::Status SetLatentAttentionParam(
    atb_speed::deepseekV2::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
    const glm::MoeLayerParam &param)
{
    SetRmsNormParam(latentAttentionParam, param);
    SetRmsNormQuantParam(latentAttentionParam, param);

    latentAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    latentAttentionParam.isBF16 = param.isBF16;
    latentAttentionParam.attnLinearQuantType = param.attnLinearQuantType;
    latentAttentionParam.packQuantType = param.packQuantType.at(0);
    latentAttentionParam.quantGroupSize = param.quantGroupSize;
    latentAttentionParam.attnLinearTransposeType = param.attnLinearTransposeType;
    latentAttentionParam.enableLcoc = param.enableLcoc;
    latentAttentionParam.qLoraRank = param.qLoraRank;
    latentAttentionParam.headNum = param.headNum;
    latentAttentionParam.qkNopeHeadDim = param.qkNopeHeadDim;
    latentAttentionParam.qkRopeHeadDim = param.qkRopeHeadDim;
    latentAttentionParam.kvLoraRank = param.kvLoraRank;
    latentAttentionParam.ropeTrans = true;
    latentAttentionParam.rotaryType = atb_speed::common::RotaryType::ALL_ROTARY;
    latentAttentionParam.ropeParam.rotaryCoeff = 2; // 2:translated
    latentAttentionParam.isFA = param.isFA;
    latentAttentionParam.isPrefill = param.isPrefill;
    latentAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    latentAttentionParam.index_head_dim = param.index_head_dim;
    latentAttentionParam.index_n_heads = param.index_n_heads;
    latentAttentionParam.index_topk = param.index_topk;
    latentAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    latentAttentionParam.selfAttentionParam.kvHeadNum = param.numAttentionHeadsPerRank;
    CHECK_PARAM_GT(param.hiddenSizePerAttentionHead, 0);
    latentAttentionParam.selfAttentionParam.qkScale = param.softmaxScale;
    latentAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
    if (param.isFA) {
        latentAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        latentAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    latentAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    latentAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    latentAttentionParam.pageAttentionParam.kvHeadNum = 1;
    latentAttentionParam.pageAttentionParam.mlaVHeadSize = param.kvLoraRank;
    latentAttentionParam.pageAttentionParam.qkScale = param.softmaxScale;
    latentAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::UNDEFINED;
    latentAttentionParam.enableMlaPreprocess = param.enableMlaPreprocess;
    latentAttentionParam.enableCustomizeMla = param.enableCustomizeMla;
    if (param.enableSpeculate) {
        if (param.maskfree) {
            latentAttentionParam.pageAttentionParam.maskType = \
                atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM;
        } else {
            latentAttentionParam.pageAttentionParam.maskType = \
                atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC;
        }
        latentAttentionParam.pageAttentionParam.calcType = atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
    }
    if (param.enableFA3 && param.enableKvQuantLayer) {
        latentAttentionParam.reshapeCacheParm.kvCacheCfg = \
            atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ;
        latentAttentionParam.pageAttentionParam.quantType = atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_ONLINE;
        latentAttentionParam.pageAttentionParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    } else if (param.isNzCache) {
        latentAttentionParam.reshapeCacheParm.kvCacheCfg = \
            atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ;
    } else {
        latentAttentionParam.reshapeCacheParm.kvCacheCfg = \
            atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE;
    }
    latentAttentionParam.isNzCache = param.isNzCache;
    // This function must be called after the pageAttentionParam is set. It will change pageAttentionParam.headNum
    SetAttnCpParam(latentAttentionParam, param);
    SetAttnInnerSpParam(latentAttentionParam, param);
    SetLatentAttentionInnerCommParam(latentAttentionParam, param);
    latentAttentionParam.enableQkvdownDp = param.enableQkvdownDp && param.layerId > param.firstKDenseReplace;
    latentAttentionParam.layerId = param.layerId;
    latentAttentionParam.firstKDenseReplace = param.firstKDenseReplace;
    latentAttentionParam.hasAttnComm = param.hasAttnComm;
    latentAttentionParam.attnTpRank = param.mapping.Get(base::ATTN_TP).rank;
    latentAttentionParam.attnTpSize = param.mapping.Get(base::ATTN_TP).rankIds.size();
    latentAttentionParam.attnTpBackend = param.mapping.Get(base::ATTN_TP).defaultBackend;
    latentAttentionParam.attnTpRankTableFile = "";
    param.mapping.Get(base::ATTN_TP).InitCommDomain(latentAttentionParam.hcclComm, latentAttentionParam.attnTpDomain);
    latentAttentionParam.ffnAllGather = param.ffnAllGather;
    latentAttentionParam.hasFfnComm = param.hasFfnComm;

    latentAttentionParam.enableOutLcocTp = param.enableOutLcocTp;
    latentAttentionParam.enablePreprocessLcocTp = param.enablePreprocessLcocTp;
    latentAttentionParam.lcocAttnTpRank = param.mapping.Get(base::ATTN_TP).rank;
    latentAttentionParam.lcocAttnTpRankSize = param.mapping.Get(base::ATTN_TP).rankIds.size();
    latentAttentionParam.lcocAttnTpBackend = "lcoc";
    param.mapping.Get(base::ATTN_TP).InitCommDomain(
        latentAttentionParam.lcocHcclComm, latentAttentionParam.lcocAttnTpDomain);
    latentAttentionParam.enablePrefixCache = param.enablePrefixCache;
    latentAttentionParam.normEps = param.normEps;
    latentAttentionParam.softmaxScale = param.softmaxScale;
    return atb::NO_ERROR;
}


template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddFusionAttention(bool is_auxiliary, uint64_t stream_id)
{
    atb::Node attentionNode;
    CHECK_OPERATION_STATUS_RETURN(this->CreateFusionAttentionOperation(&attentionNode.operation));

    if (this->param.useMLA){
        
        std::string tmp = is_auxiliary ? "in_hidden_states_auxiliary" : "in_hidden_states";
        std::vector<std::string> attnInTensorNames = {tmp,
            "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias"
        };
        
        atb_speed::common::AddTensorToList(LayerAttnInTensorDefaultCandidates(), "in_attn_weight", attnInTensorNames);

        // TODO - DSA
        // if (this->param.index_n_heads > 0) {
        //     atb_speed::common::AddTensorToList(LayerAttnInTensorDefaultCandidates(), "indexer_weight", attnInTensorNames);
        // }

        // changed in_v_cache -> in_k_rope_cache
        // ADD in_q_len
        // ADD in_attn_padding_idx for OprojTp
        janus::atb_utils::append(attnInTensorNames, (is_auxiliary ? 
            (std::vector<std::string>{"in_cos_embedding_auxiliary", "in_sin_embedding_auxiliary", "in_seq_len_auxiliary", "in_k_cache_auxiliary", "in_v_cache_auxiliary",
            "in_attention_mask_auxiliary", "in_q_len_auxiliary", "in_token_offset_auxiliary", "in_layer_id_auxiliary", "in_block_tables_auxiliary",
            "in_slots_auxiliary", "in_attn_padding_idx_auxiliary"}) : 
            (std::vector<std::string>{"in_cos_embedding", "in_sin_embedding", "in_seq_len", "in_k_cache", "in_k_rope_cache", // in_k_rope_cache
            "in_attention_mask", "in_q_len", "in_token_offset", "in_layer_id", "in_block_tables",
            "in_slots", "in_attn_padding_idx"})));
        // if (param.index_n_heads > 0) {
        //     atb_speed::common::AddTensorToList(LayerAttnInTensorDefaultCandidates(), "indexer_intensor", attnInTensorNames);
        // }
        if (this->param.enablePrefixCache) {
            atb_speed::common::AddTensorToList(LayerAttnInTensorDefaultCandidates(), "prefixcache", attnInTensorNames);
        }

        // TODO
        // if (this->param.enableFA3 && this->param.enableKvQuantLayer) {
        //     atb_speed::common::AddTensorToList(LayerAttnInTensorDefaultCandidates(), "fa3_quant", attnInTensorNames);
        // }
        // if (this->param.enableQkvdownDp && this->param.layerId > this->param.firstKDenseReplace) {
        //     // h3p qkvdown dptranslatedsptranslated,translatedlatent_attention.cpptranslatedinTensortranslated,
        //     // h3p qkvdown dptranslatedinTensortranslatedsptranslatedinTensortranslated
        //     attnInTensorNames.push_back(is_auxiliary ? "in_ffn_unpadding_idx_auxiliary" : "in_ffn_unpadding_idx");
        // }

        // TODO CP+SP
        // if (this->param.mapping.Get(base::ATTN_CP).IsEnabled() && this->param.isPrefill) {
        //     attnInTensorNames.push_back("in_seq_len_cp");
        //     attnInTensorNames.push_back("in_cp_load_balance_idx_first");
        //     attnInTensorNames.push_back("in_cp_load_balance_idx_last");
        //     attnInTensorNames.push_back("in_cp_o_recover_idx");
        //     attnInTensorNames.push_back("in_cp_kv_recover_idx");
        // }
        // if (this->param.mapping.Get(base::ATTN_INNER_SP).IsEnabled() && !this->param.isPrefill) {
        //     attnInTensorNames.push_back("in_seq_len_sp");
        // }
        attentionNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, attnInTensorNames);
        attentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {is_auxiliary ? 
            "intermediate_attn_out_auxiliary" : "intermediate_attn_out"});
    }
    else {
        // translatedtensortranslated
        std::map<unsigned int, std::vector<std::string>> attnInTensor = this->GetAttentionIntensor();
        std::vector<std::string> attnInTensorNames = {};
        attnInTensorNames.reserve(attnInTensor.size());

        // TODO change input
        for (unsigned int i = 0; i < common::AttnInTensorCategory::ATTN_END; i++) {
            if (!is_auxiliary) {
                if (i == common::AttnInTensorCategory::ATTN_SPECULATE && this->param.isPrefill) {
                    if (attnInTensor[i].size() > 0) {
                        attnInTensorNames.push_back(attnInTensor[i][0]);
                        continue;
                    }
                }
                attnInTensorNames.insert(attnInTensorNames.end(), attnInTensor[i].begin(), attnInTensor[i].end());
            } else {
                if (i == common::AttnInTensorCategory::ATTN_SPECULATE && this->param.isPrefill) {
                    if (attnInTensor[i].size() > 0) {
                        attnInTensorNames.push_back(attnInTensor[i][1]);
                        continue;
                    }
                }
                std::transform(attnInTensor[i].begin(), attnInTensor[i].end(), std::back_inserter(attnInTensorNames),
                    [this](const std::string& str) {
                        auto isNotDuplicated = std::find(this->inTensorCandidates["default"].begin(), 
                            this->inTensorCandidates["default"].end(), str) == this->inTensorCandidates["default"].end();
                        return isNotDuplicated ? str : (str + ("_auxiliary"));
                    });
            }
        }

        attentionNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, attnInTensorNames);
        std::vector<std::string> attnOutTensorName = {is_auxiliary ? "intermediate_attn_out_auxiliary" : "intermediate_attn_out"};
        if (this->param.enableInterLayerAddNorm && (this->param.layerId != 0)) {
            attnOutTensorName.push_back(is_auxiliary ? "in_hidden_states_auxiliary" : "in_hidden_states");
        }
        attentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, attnOutTensorName);
    }

    if (stream_id) {
        atb::SetExecuteStreamId(attentionNode.operation, stream_id);
    }

    this->graph.nodes.push_back(attentionNode);

    // removed prefetch code

    return atb::NO_ERROR;
}


template <>
atb::Status MoeDecoderLayer<atb::infer::RmsNormParam>::CreateFusionAttentionOperation(atb::Operation **op)
{
    if (this->param.useMLA){
        atb_speed::deepseekV2::LatentAttentionParam<atb::infer::RmsNormParam> latentAttentionParam;
        SetLatentAttentionParam(latentAttentionParam, this->param);
        CHECK_OPERATION_STATUS_RETURN(atb_speed::deepseekV2::Attention(latentAttentionParam, op));
    }
    else {
        atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
        this->SetFusionAttentionParam(fusionAttentionParam);
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::Attention(fusionAttentionParam, op));
    }

    return atb::NO_ERROR;
}

template <>
atb::Status MoeDecoderLayer<atb::infer::LayerNormParam>::CreateFusionAttentionOperation(atb::Operation **op)
{
    atb_speed::common::FusionAttentionParam<atb::infer::LayerNormParam> fusionAttentionParam;
    this->SetFusionAttentionParam(fusionAttentionParam);
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::Attention(fusionAttentionParam, op));
    return atb::NO_ERROR;
}


template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::BuildGraph(atb::Operation **operation)
{

    this->param.enableIntraLayerAddNorm = true;
    this->param.enableFusedTopk = true;
    this->param.CalculateDataPartition();
    this->param.CalculateCommType();

    moe::MoeDecoderLayer<NormType>::BuildGraph(operation);

    return atb::NO_ERROR;
}


template class MoeDecoderLayer<atb::infer::RmsNormParam>;
template class MoeDecoderLayer<atb::infer::LayerNormParam>;

} // namespace glm
} // namespace atb_speed
