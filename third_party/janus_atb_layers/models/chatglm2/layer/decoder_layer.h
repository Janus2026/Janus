/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except inccompliance with the License.
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
#ifndef ATB_SPEED_MODELS_GLM_DECODER_LAYER_H
#define ATB_SPEED_MODELS_GLM_DECODER_LAYER_H

#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace chatglm {
struct DecoderLayerParam {
    bool isFA = true;
    bool isPrefill = false;
    bool isBF16 = false;
    bool isPack = true;
    bool supportSwiGLU = false;
    bool supportSpeculate = false;
    bool kvQuant = false;
    bool supportCompressHead = false;

    int quantType = 0;
    float rmsNormEps = 0;
    int numAttentionHeadsPerRank = 0;
    int hiddenSizePerAttentionHead = 0;
    int numKeyValueHeadsPerRank = 0;
    int rank = 0;
    int worldSize = 1;
    std::string backend = "hccl";
    std::vector<int> seqLen;
    std::vector<int> tokenOffset;
    float preScale = 0;
    float postScale = 0;
    std::string weightQuantType = "float";
    std::vector<int> packQuantType = {};  // translated,translatedQKV packtranslated,translatedMLP packtranslated
    // translated,translatedq,k,v,self attention out,gate,up,down lineartranslated
    std::vector<int> linearQuantType = {};
    std::vector<int> linearTransposeType;
};

enum DecoderLayerTensorIdx : uint32_t {
    // shape: FA: [batchSize, seqLen, maxPositionEmbeddings] PA: [seqLen, hiddenSize]
    IN_HIDDEN_STATES = 0,
    IN_INPUT_NORM_WEIGHT,               // shape: [hiddenSize]
    IN_INPUT_NORM_BIAS,
    IN_INPUT_NORM_NEW_WEIGHT,
    IN_INPUT_NORM_NEW_BIAS,
    // Pack:
    // MHA [3 * numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    // GQA [(numAttentionHeadsPerRank + 2 * numKeyValueHeadsPerRank) * hiddenSizePerAttentionHead, hiddenSize]
    // No pack:
    // (Q) shape: [numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_WEIGHT_0,
    IN_QKV_DEOFFSET_0,                  // Quanttranslated
    IN_QKV_DESCALE_0,                   // Quanttranslated
    IN_QKV_OFFSET_0,                    // Quanttranslated
    IN_QKV_SCALE_0,                     // Quanttranslated
    IN_QKV_COMPRESS_IDX_0,              // Quanttranslated
    // Pack: no usage; No pack: (K) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_WEIGHT_1,
    IN_QKV_DEOFFSET_1,                  // Quanttranslated
    IN_QKV_DESCALE_1,                   // Quanttranslated
    IN_QKV_OFFSET_1,                    // Quanttranslated
    IN_QKV_SCALE_1,                     // Quanttranslated
    IN_QKV_COMPRESS_IDX_1,              // Quanttranslated
    // Pack: no usage; No pack: (V) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_WEIGHT_2,
    IN_QKV_DEOFFSET_2,                  // Quanttranslated
    IN_QKV_DESCALE_2,                   // Quanttranslated
    IN_QKV_OFFSET_2,                    // Quanttranslated
    IN_QKV_SCALE_2,                     // Quanttranslated
    IN_QKV_COMPRESS_IDX_2,              // Quanttranslated
    // shape: [hiddenSize, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    IN_ATTENTION_OUT_WEIGHT,
    IN_ATTENTION_OUT_DEOFFSET,          // Quanttranslated
    IN_ATTENTION_OUT_DESCALE,           // Quanttranslated
    IN_ATTENTION_OUT_OFFSET,            // Quanttranslated
    IN_ATTENTION_OUT_SCALE,             // Quanttranslated
    IN_ATTENTION_OUT_COMPRESS_IDX,      // Quanttranslated
    IN_ATTENTION_NORM_WEIGHT,           // shape: [hiddenSize]
    IN_ATTENTION_NORM_BIAS,
    IN_ATTENTION_NORM_NEW_WEIGHT,
    IN_ATTENTION_NORM_NEW_BIAS,
    // Pack: shape: [2 * intermediateSizePerRank, hiddenSize]
    // No pack: (Gate) shape: [intermediateSizePerRank, hiddenSize]
    IN_MLP_WEIGHT_0,
    IN_MLP_DEOFFSET_0,                  // Quanttranslated
    IN_MLP_DESCALE_0,                   // Quanttranslated
    IN_MLP_OFFSET_0,                    // Quanttranslated
    IN_MLP_SCALE_0,                     // Quanttranslated
    IN_MLP_COMPRESS_IDX_0,
    // Pack: no usage; No pack: (Up) shape: [intermediateSizePerRank, hiddenSize]
    IN_MLP_WEIGHT_1,
    IN_MLP_DEOFFSET_1,                  // Quanttranslated
    IN_MLP_DESCALE_1,                   // Quanttranslated
    IN_MLP_OFFSET_1,                    // Quanttranslated
    IN_MLP_SCALE_1,                     // Quanttranslated
    IN_MLP_COMPRESS_IDX_1,
    IN_MLP_DOWN_WEIGHT,                 // shape: [hiddenSize, intermediateSizePerRank]
    IN_MLP_DOWN_DEOFFSET,               // Quanttranslated
    IN_MLP_DOWN_DESCALE,                // Quanttranslated
    IN_MLP_DOWN_OFFSET,                 // Quanttranslated
    IN_MLP_DOWN_SCALE,                  // Quanttranslated
    IN_MLP_DOWN_COMPRESS_IDX,           // Quanttranslated

    IN_K_QUANT_SCALE,                  // kv quant
    IN_K_DEQUANT_SCALE,                // kv quant
    IN_V_QUANT_SCALE,                  // kv quant
    IN_V_DEQUANT_SCALE,                // kv quant
    IN_K_QUANT_OFFSET,                 // kv quant
    IN_K_DEQUANT_OFFSET,               // kv quant
    IN_V_QUANT_OFFSET,                 // kv quant
    IN_V_DEQUANT_OFFSET,               // kv quant

    // shape: FA: [batchSize * seqLen, hiddenSizePerAttentionHead] PA: [seqLen, hiddenSizePerAttentionHead]
    IN_COS_TABLE,
    // shape: FA: [batchSize * seqLen, hiddenSizePerAttentionHead] PA: [seqLen, hiddenSizePerAttentionHead]
    IN_SIN_TABLE,
    // shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings] PA: [seqLen, seqLen]
    IN_ATTENTION_MASK,
    // shape: FA: [batchSize, maxPositionEmbeddings, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead]
    // PA: [2622, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
    IN_K_CACHE,
    // shape: FA: [batchSize, maxPositionEmbeddings, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead]
    // PA: [2622, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
    IN_V_CACHE,
    IN_SEQ_LEN,                         // shape: [batchSize]
    IN_PLACE_HOLDER,                    // shape: [1]
    IN_TOKEN_OFFSET,                    // shape: [batchSize]; FAtranslated
    IN_LAYER_ID,                        // shape: [1]; FAtranslated
    IN_BLOCK_TABLES,                    // shape: [seqLen, seqLen]; PAtranslated
    IN_SLOTS,                           // shape: [seqLen]; PAtranslated
    IN_BATCH_WINS,
    IN_RA_SEQ_LEN,
    IN_PFFSET_INDEX,
    IN_RA_OFFSET,
    IN_RESHAPE_SEQ_LEN,
    IN_Q_LEN,                           // shape: [batchsize];translated

    // shape: FA: [batchSize, seqLen, maxPositionEmbeddings] PA: [seqLen, hiddenSize]
    OUT_DECODER_LAYER,

    INTERMEDIATE_ATTENTION_OUT,         // shape: PA: [seqLen, hiddenSize]
    INTERMEDIATE_MLP_OUT,               // shape: PA: [seqLen, hiddenSize]
};

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation);

}  // namespace chatglm
}  // namespace atb_speed
#endif