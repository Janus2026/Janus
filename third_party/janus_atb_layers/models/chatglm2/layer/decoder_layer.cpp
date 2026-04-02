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
#include "atb_speed/log.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/attention/fusion_attention.h"
#include "operations/fusion/mlp/mlp.h"

#include "models/chatglm2/layer/decoder_layer.h"

namespace atb_speed {
namespace chatglm {

static const std::string WEIGHT_QUANT_TPYE = "w8a8sc";
static const std::string WEIGHT_W4A16_QUANT_TPYE = "w4a16";
static const uint64_t IN_TENSOR_COUNT = 76;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 2;

int64_t SetSelfAttentionParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam,
    const DecoderLayerParam &param
)
{
    // self attention param
    fusionAttentionParam.isFA = param.isFA;
    fusionAttentionParam.isPrefill = param.isPrefill;

    fusionAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    if (param.hiddenSizePerAttentionHead < 1) {
        ATB_SPEED_LOG_ERROR("hiddenSizePerAttentionHead which is smaller than 1!");
        return atb::ERROR_INVALID_GRAPH;
    }
    if (fusionAttentionParam.isFA) {
        fusionAttentionParam.selfAttentionParam.qScale = param.preScale;
        fusionAttentionParam.selfAttentionParam.qkScale = param.postScale;
        fusionAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        fusionAttentionParam.selfAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }

    fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;

    fusionAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;

    fusionAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::UNDEFINED;

    if (param.supportSpeculate) {
        fusionAttentionParam.pageAttentionParam.calcType = \
            atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
        fusionAttentionParam.pageAttentionParam.maskType = \
            atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC;
    }

    if (param.supportCompressHead) {
        fusionAttentionParam.pageAttentionParam.compressType =
            atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE;
        fusionAttentionParam.reshapeCacheParm.compressType =
            atb::infer::ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE;
    }

    return atb::NO_ERROR;
}

int64_t SetFusionAttentionParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam,
    const DecoderLayerParam &param
)
{
    // QKV linear param
    fusionAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.isBF16 = param.isBF16;
    fusionAttentionParam.packQuantType = param.packQuantType.at(0);
    fusionAttentionParam.layerLinearQuantType = param.linearQuantType;
    fusionAttentionParam.layerLinearTransposeType = param.linearTransposeType;
    fusionAttentionParam.quantGroupSize = 128;  // 128: w4a16 pre grouptranslatedgroup size
    fusionAttentionParam.qkvHasBias = true;

    // rmsNorm param
    atb::infer::RmsNormParam attenRmsNormParam;
    attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormParam.normParam.epsilon = param.rmsNormEps;
    fusionAttentionParam.normParamType = attenRmsNormParam;
    atb::infer::RmsNormParam attenRmsNormQuantParam;
    attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
    attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    fusionAttentionParam.normQuantParamType = attenRmsNormQuantParam;
    fusionAttentionParam.enableNormQuantOp = false;

    // rope param
    fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::HALF_ROTARY;
    fusionAttentionParam.ropeParam.rotaryCoeff = param.hiddenSizePerAttentionHead / 2;  // translated headdim/2

    CHECK_OPERATION_STATUS_RETURN(SetSelfAttentionParam(fusionAttentionParam, param));

    // self out linear param
    if (param.weightQuantType.compare(WEIGHT_QUANT_TPYE) == 0) {
        fusionAttentionParam.denseQuantType = atb_speed::common::ALL_W8A8SC;
    } else if (param.weightQuantType.compare(WEIGHT_W4A16_QUANT_TPYE) == 0) {
        fusionAttentionParam.denseQuantType = atb_speed::common::ALL_W4A16;
    }
    fusionAttentionParam.selfOutLinearTensorParallelInfo = {param.rank, param.worldSize, param.backend};

    if (param.kvQuant) {
        fusionAttentionParam.pageAttentionParam.quantType = atb::infer::PagedAttentionParam::TYPE_DEQUANT_FUSION;
        fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::UNDEFINED;
        fusionAttentionParam.pageAttentionParam.hasQuantOffset  = true;
    }

    return atb::NO_ERROR;
}

int64_t AddFusionAttention(atb::Node &attentionNode, const DecoderLayerParam &param)
{
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    CHECK_OPERATION_STATUS_RETURN(SetFusionAttentionParam(fusionAttentionParam, param));
    CHECK_OPERATION_STATUS_RETURN(Attention(fusionAttentionParam, &attentionNode.operation));
    attentionNode.inTensorIds = {
        IN_HIDDEN_STATES, IN_INPUT_NORM_WEIGHT, IN_INPUT_NORM_BIAS, IN_INPUT_NORM_NEW_WEIGHT, IN_INPUT_NORM_NEW_BIAS,
        IN_QKV_WEIGHT_0, IN_QKV_SCALE_0, IN_QKV_OFFSET_0, IN_QKV_DESCALE_0, IN_QKV_DEOFFSET_0, IN_QKV_COMPRESS_IDX_0,
        IN_QKV_WEIGHT_1, IN_QKV_SCALE_1, IN_QKV_OFFSET_1, IN_QKV_DESCALE_1, IN_QKV_DEOFFSET_1, IN_QKV_COMPRESS_IDX_1,
        IN_QKV_WEIGHT_2, IN_QKV_SCALE_2, IN_QKV_OFFSET_2, IN_QKV_DESCALE_2, IN_QKV_DEOFFSET_2, IN_QKV_COMPRESS_IDX_2,
        IN_COS_TABLE, IN_SIN_TABLE, IN_SEQ_LEN, IN_K_CACHE, IN_V_CACHE,
        IN_ATTENTION_MASK, IN_TOKEN_OFFSET, IN_LAYER_ID, IN_BLOCK_TABLES, IN_SLOTS,
        IN_ATTENTION_OUT_WEIGHT, IN_ATTENTION_OUT_SCALE, IN_ATTENTION_OUT_OFFSET,
        IN_ATTENTION_OUT_DESCALE, IN_ATTENTION_OUT_DEOFFSET, IN_ATTENTION_OUT_COMPRESS_IDX
    };
    if (param.supportCompressHead) {
        attentionNode.inTensorIds.push_back(IN_BATCH_WINS);
        attentionNode.inTensorIds.push_back(IN_RA_SEQ_LEN);
        attentionNode.inTensorIds.push_back(IN_PFFSET_INDEX);
        attentionNode.inTensorIds.push_back(IN_RA_OFFSET);
        attentionNode.inTensorIds.push_back(IN_RESHAPE_SEQ_LEN);
    }
    if (param.supportSpeculate) {
        attentionNode.inTensorIds.push_back(IN_Q_LEN);
    }
    if (param.kvQuant) {
        std::vector<uint32_t> kvQuantInTensorIds = {
            IN_K_QUANT_SCALE, IN_K_DEQUANT_SCALE, IN_V_QUANT_SCALE, IN_V_DEQUANT_SCALE,
            IN_K_QUANT_OFFSET, IN_K_DEQUANT_OFFSET, IN_V_QUANT_OFFSET, IN_V_DEQUANT_OFFSET,
        };
        for (auto tensorIds : kvQuantInTensorIds) {
            attentionNode.inTensorIds.push_back(tensorIds);
        }
    }

    attentionNode.outTensorIds = {INTERMEDIATE_ATTENTION_OUT};

    return atb::NO_ERROR;
}

int64_t AddMlp(atb::Node &mlpParallelNode, const DecoderLayerParam &param)
{
    atb_speed::common::MlpParam<atb::infer::RmsNormParam> mlpParam;
    mlpParam.isBF16 = param.isBF16;
    mlpParam.packQuantType = param.packQuantType.at(1);
    mlpParam.layerLinearQuantType = param.linearQuantType;
    mlpParam.layerLinearTransposeType = param.linearTransposeType;
    mlpParam.quantGroupSize = 128;  // 128: w4a16 pre grouptranslatedgroup size
    // gate up
    mlpParam.mlpPackType = atb_speed::common::GetMlpPackType(param.packQuantType.at(1), false);
    atb::infer::RmsNormParam mlpRmsNormParam;
    mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    mlpRmsNormParam.normParam.epsilon = param.rmsNormEps;
    mlpParam.normParamType = mlpRmsNormParam;
    atb::infer::RmsNormParam mlpRmsNormQuantParam;
    mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    mlpRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
    mlpRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.normQuantParamType = mlpRmsNormQuantParam;
    if (param.weightQuantType.compare(WEIGHT_QUANT_TPYE) == 0) {
        mlpParam.downQuantType = atb_speed::common::ALL_W8A8SC;
    } else if (param.weightQuantType.compare(WEIGHT_W4A16_QUANT_TPYE) == 0) {
        mlpParam.downQuantType = atb_speed::common::ALL_W4A16;
    }
    mlpParam.downLinearTensorParallelInfo = {param.rank, param.worldSize, param.backend};
    if (param.supportSwiGLU) {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
        mlpParam.activationParam.dim = -1;
        CHECK_OPERATION_STATUS_RETURN(MlpSwiGLU(mlpParam, &mlpParallelNode.operation));
    } else {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
        CHECK_OPERATION_STATUS_RETURN(Mlp(mlpParam, &mlpParallelNode.operation));
    }

    mlpParallelNode.inTensorIds = {
        IN_HIDDEN_STATES, IN_ATTENTION_NORM_WEIGHT, IN_ATTENTION_NORM_BIAS,
        IN_ATTENTION_NORM_NEW_WEIGHT, IN_ATTENTION_NORM_NEW_BIAS,
        IN_MLP_WEIGHT_0, IN_MLP_SCALE_0, IN_MLP_OFFSET_0, IN_MLP_DESCALE_0, IN_MLP_DEOFFSET_0, IN_MLP_COMPRESS_IDX_0,
        IN_MLP_WEIGHT_1, IN_MLP_SCALE_1, IN_MLP_OFFSET_1, IN_MLP_DESCALE_1, IN_MLP_DEOFFSET_1, IN_MLP_COMPRESS_IDX_1,
        IN_MLP_DOWN_WEIGHT, IN_MLP_DOWN_SCALE, IN_MLP_DOWN_OFFSET, IN_MLP_DOWN_DESCALE,
        IN_MLP_DOWN_DEOFFSET, IN_MLP_DOWN_COMPRESS_IDX
    };
    mlpParallelNode.outTensorIds = {INTERMEDIATE_MLP_OUT};

    return atb::NO_ERROR;
}

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";

    atb::Node attentionNode;
    atb::Node selfResidualAddNode;
    atb::Node mlpParallelNode;
    atb::Node mlpResidualAddNode;

    CHECK_OPERATION_STATUS_RETURN(AddFusionAttention(attentionNode, param));
    opGraph.nodes.push_back(attentionNode);

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {
        IN_HIDDEN_STATES,
        INTERMEDIATE_ATTENTION_OUT
    };
    selfResidualAddNode.outTensorIds = {IN_HIDDEN_STATES};
    opGraph.nodes.push_back(selfResidualAddNode);

    CHECK_OPERATION_STATUS_RETURN(AddMlp(mlpParallelNode, param));
    opGraph.nodes.push_back(mlpParallelNode);

    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {
        IN_HIDDEN_STATES,
        INTERMEDIATE_MLP_OUT
    };
    mlpResidualAddNode.outTensorIds = {OUT_DECODER_LAYER};
    opGraph.nodes.push_back(mlpResidualAddNode);

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

} // namespace chatglm
} // namespace atb_speed