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
#ifndef ATB_SPEED_BASE_PARAM_H
#define ATB_SPEED_BASE_PARAM_H
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"
#include "operations/fusion/utils.h"
#include "models/base/param/mapping.h"

namespace atb_speed {
namespace base {

/// An enumerator specifies various types of position embeddings
enum PositionEmbeddingType : uint32_t {
    /// Rotary position embedding
    ROPE = 0,
    /// Attention with linear biases
    ALIBI,
    /// Absolute position encodings
    ABSOLUTE,
};

/// An enumerator specifies various types of layer normalization
enum NormType : uint32_t {
    /// Root mean square normalization
    RMS_NORM = 0,
    /// Layer normalization
    LAYER_NORM,
};

/// An enumerator represents the positional index of linear bias
enum HasBias : uint32_t {
    /// the positional index of the bias in the QKV linear
    QKV_HASBIAS = 0,
    /// the positional index of the bias in the attention's dense linear
    SELFATTENTION_HASBIAS,
    /// the positional index of the bias in the gate up linear
    GATEUP_HASBIAS,
    /// the positional index of the bias in the mlp's down linear
    DOWN_HASBIAS,
};

/// Common parameters shared between the model and layer classes
class Param {
public:
    /// If `isFA` is true, Flash Attention is used; otherwise, Paged Attention is used
    bool isFA = true;
    /// A flag that indicates whether the input includes padding
    bool isUnpadInputs = true;
    /// A flag indicating the prefill and decode phases
    bool isPrefill = false;
    /// When `isBF16` is true, bfloat16 precision is used; otherwise, float16 precision is used.
    bool isBF16 = false;
    /// A flag indicating that an edge device is used
    bool isEdgeHardware = false;
    /// A flag that indicates whether the MLP module utilizes the SwiGLU fusion operation
    bool enableSwiGLU = false;
    /// A flag indicating whether q_norm and k_norm is enabled
    bool useQKNorm = false;

    bool usePostSelfAttnLayerNorm = false;
    bool usePostMlpLayerNorm = false;
    /// A flag that indicates whether the shared exports module utilizes the SwiGLUQuant fusion operation
    bool enableSwiGLUQuantForSharedExperts = false;
    /// A flag that indicates whether low-latency computation over communication is enabled
    bool enableLcoc = false;
    // A flag that indicates whether mc2 is enabled
    bool enableMC2 = false;
    /// A flag indicating whether speculation is enabled
    bool enableSpeculate = false;
    /// A flag indicating whether razor attention is enabled
    bool enableCompressHead = false;
    /// A flag indicating whether omni attention is enabled
    bool enableOmniAttention = false;
    bool isomnicompressed = false;
    /// A vector stores compressed info of each layer
    std::vector<bool> patternMask = {};
    /// A flag indicating whether split fuse is enabled
    bool enableSplitFuse = false;
    /// A flag indicating whether lora is enabled
    bool enableLora = false;
    /// A flag indicating whether the group matmul operation is enabled;
    /// it should be activated when batch inputs include multiple LoRA adapters
    bool loraEnableGMM = false;
    /// A flag indicating whether to use int8 quantization for the KV cache
    bool enableKvQuant = false;
    /// A flag indicating whether to use int8 quantization for the KV cache layer
    bool enableKvQuantLayer = false;
    /// A flag indicating whether int8 quantization for the KV cache has offset (i.e., asymmetric)
    bool kvQuantHasOffset = true;
    /// A flag indicating whether RopeQuantKvcache is enabled (i.e., asymmetric)
    bool enableRopeQuantKvcache = false;
    /// A flag indicating whether q_norm and k_norm is rmsnorm; maybe should use normType
    bool rmsnormQKNorm = false;
    /// A flag indicating whether flash attention 3 is enabled
    bool enableFA3 = false;
    /// A flag indicating whether all reduce quantization is enabled.
    /// It can be enabled only when the communication backend is set to "lccl".
    bool enableReduceQuant = false;
    /// Whether to enable inter-layer addRmsNorm fusion, default false.
    bool enableInterLayerAddNorm = false;
    /// Whether to enable intra-layer addRmsNorm fusion, default false.
    bool enableIntraLayerAddNorm = false;
    /// A flag indicating whether prefix cache is enabled
    bool enablePrefixCache = false;
    /// A flag indicating whether prefetch weight
    bool enablePreFetchWeight = false;
    /// A flag indicating whether the model use cube and vector parallel
    bool enableCVOverlap = false;
    /// A flag indicating whether the attention module is skipped in the layer
    bool isAttnSkipLayer = false;
    /// A flag indicating whether the mlp module is skipped in the layer
    bool isMlpSkipLayer = false;
    /// A flag indicating whether to use swigluQuant
    bool enableSwigluQuant = false;
    /// A flag indicating whether open flashcomm1.0
    bool enableFlashComm = false;
    /// A flag indicating whether to use pmcc model obfuscation
    bool enableModelConfuscation = false;
    /// A handle used by pmcc model obfuscation
    int32_t modelConfuscationFd = 0;
    /// The backend of the attention module; refer to `OpBackend` for the supported values
    atb_speed::common::OpBackend attnBackend = atb_speed::common::OpBackend::ATB;
    /// The backend of the matmul module; refer to `OpBackend` for the supported values
    atb_speed::common::OpBackend matmulBackend = atb_speed::common::OpBackend::ATB;
    /// The type of the position embedding; refer to `PositionEmbeddingType` for the supported values
    PositionEmbeddingType positionEmbeddingType = PositionEmbeddingType::ROPE;
    /// The epsilon value used for normalization
    float normEps = 0;
    /// The type of the normalization; refer to `NormType` for the supported values
    NormType normType = NormType::RMS_NORM;
    /// The group size used for dequantizing the weight tensor in the per-group quantization approach
    uint32_t quantGroupSize = 0;
    /// Number of attention heads per rank
    uint32_t numAttentionHeadsPerRank = 0;
    /// The dimension of the hidden representations for each attention head
    uint32_t hiddenSizePerAttentionHead = 0;
    /// If `numKeyValueHeadsPerRank` equals to `numAttentionHeadsPerRank`, the model will use Multi Head Attention;
    /// otherwise, Grouped Query Attention is used
    uint32_t numKeyValueHeadsPerRank = 0;
    /// The quantization type applied to the model
    std::string weightQuantType = "";
    // Model Parallelism
    Mapping mapping;
    std::string backend = "hccl";
    bool hasAttnTp = false;
    int attnTpRank = 0;
    int attnTpSize = 1;
    std::string attnTpDomain = "";
    std::string attnTpRankTableFile = "";
    std::string attnTpBackend = "";
    bool hasAttnDp = false;
    int attnDpRank = 0;
    int attnDpSize = 1;
    std::string attnDpDomain = "";
    std::string attnDpRankTableFile = "";
    std::string attnDpBackend = "";

    bool hasMlpEp = false;
    int mlpEpRank = 0;
    int mlpEpSize = 1;
    std::string mlpEpDomain = "";
    std::string mlpEpRankTableFile = "";
    std::string mlpEpBackend = "";
    bool hasMlpTp = false;
    int mlpTpRank = 0;
    int mlpTpSize = 1;
    std::string mlpTpDomain = "";
    std::string mlpTpRankTableFile = "";
    std::string mlpTpBackend = "";

    bool enableAclGraphPagedAttention = false;

    Param() {};
    virtual ~Param() {};

    /// A member function that outputs the values of all parameters
    virtual void PrintParam()
    {
        ATB_SPEED_LOG_DEBUG("Param: " << "isFA: " << isFA
                      << ", isUnpadInputs: " << isUnpadInputs
                      << ", isPrefill: " << isPrefill
                      << ", isBF16: " << isBF16
                      << ", isEdgeHardware: " << isEdgeHardware
                      << ", enableSwiGLU: " << enableSwiGLU
                      << ", enableLcoc: " << enableLcoc
                      << ", enableSpeculate: " << enableSpeculate
                      << ", enableCompressHead: " << enableCompressHead
                      << ", enableOmniAttention: " << enableOmniAttention
                      << ", enableSplitFuse: " << enableSplitFuse
                      << ", enableLora: " << enableLora
                      << ", useQKNorm: " << useQKNorm
                      << ", loraEnableGMM: " << loraEnableGMM
                      << ", enableKvQuant: " << enableKvQuant
                      << ", enableReduceQuant: " << enableReduceQuant
                      << ", enableInterLayerAddNorm: " << enableInterLayerAddNorm
                      << ", enableIntraLayerAddNorm: " << enableIntraLayerAddNorm
                      << ", enablePrefixCache: " << enablePrefixCache
                      << ", attnBackend: " << attnBackend
                      << ", positionEmbeddingType: " << positionEmbeddingType
                      << ", normType: " << normType
                      << ", normEps: " << normEps
                      << ", quantGroupSize: " << quantGroupSize
                      << ", numAttentionHeadsPerRank: " << numAttentionHeadsPerRank
                      << ", hiddenSizePerAttentionHead: " << hiddenSizePerAttentionHead
                      << ", numKeyValueHeadsPerRank: " << numKeyValueHeadsPerRank
                      << ", enableMC2: " << enableMC2
                      << ", weightQuantType: " << weightQuantType
                      << ", enableSwigluQuant" << enableSwigluQuant
                      << ", matmulBackend" << matmulBackend);
    }
    /// A member function that checks and validates the values of all parameters
    virtual void CheckParam() {};
};
} // namespace base
} // namespace atb_speed


#endif