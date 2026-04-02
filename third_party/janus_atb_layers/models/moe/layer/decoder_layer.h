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
#ifndef ATB_SPEED_MODELS_MOE_DECODER_LAYER_H
#define ATB_SPEED_MODELS_MOE_DECODER_LAYER_H

#include <vector>
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "operations/fusion/moe/sparse_moe.h"
#include "operations/fusion/moe/moe_shared_expert.h"
#include "models/base/param/layer_param.h"
#include "models/base/layer/decoder_layer.h"

namespace atb_speed {
namespace moe {

/// A class that defines the parameters of base MoE layers.
class MoeLayerParam : public atb_speed::base::LayerParam {
public:
    MoeLayerParam() = default;
    virtual ~MoeLayerParam() override = default;
    void PrintParam() override;

    MoeLayerParam(const MoeLayerParam&) = default;
    MoeLayerParam& operator=(const MoeLayerParam&) = default;

    MoeLayerParam(MoeLayerParam&&) = default;
    MoeLayerParam& operator=(MoeLayerParam&&) = default;

    atb::Status CalculateDataPartition();
    atb::Status CalculateCommType();
    /// A flag indicating whether this layer is last layer.
    bool isLastLayer = false;
    /// Deprecated.
    bool enableTopKSoftmax = false;
    /// A flag indicating whether matrecies need to be transpose for matrix multiplications.
    bool transpose = true;
    /// The total number of experts utilized by the model.
    uint32_t numOfExperts = 64;
    /// The number of experts loaded to the device.
    uint32_t numOfDeviceExperts = 64;
    /// The size of the expert parallelism communication domain.
    int expertParallelDegree = 0;
    /// The flag indicating whether dynamic EP is enable.
    bool enableDynamicEp = false;
    /// A flag that indicates whether the norm in this layer has bias.
    bool normHasBias = false;
    /// A flag indicating whether or not to use integrated routing operators.
    bool enableFusedRouting = false;
    /// A flag indicating whether or not to use integrated GMM+Swiglu+quant operators.
    bool enableGMMSwigluQuant = false;
    /// A flag indicating whether or not to use fused atb GMM+Swiglu+quant operators instead of aclnn.
    bool enableAtlasGMMFused = false;
    /// A flag indicating whether to use the integrated routing-quant operator
    bool enableInitQuant = false;
    /// A flag indicating whether to use the integrated swiglu-quant operator
    bool enableInitRoutingV3 = false; 
    /// A flag inidicating whether to use aclnnInitRoutingV2 or V3 operator
    bool enableFusedReducesumDiv = false;
    /// A flag inidicating whether to use MoeFusedReducesumDivide operator
    bool enableSwigluQuant = false;
    /// A flag indicating whether to use dispatch_v2 and combine_v2
    bool enableDispatchCombineV2 = false;
    /// The way in which the top k experts are selected.
    std::string routingMethod = "softMaxTopK";
    /// The way in which expert scores are further processed.
    std::string processLogits = "normalization";
    // ==== Param for Grouped topk ====
    int moePackQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    int scaledTopk = -1;
    /// A flag indicating whether to use scaled topk option
    bool enableInitRoutingCutoff = false;
    float routedScalingFactor = 1;
    /// A flag indicating whether or not to use fused topk.
    bool enableFusedTopk = false;
    // ==== Param for shared expert and dense layer ====

    /// A flag indicating whether this layer is a dense lyaer.
    bool isDenseLayer = false;
    /// A flag indicating whether the model uses shared experts across layers.
    bool hasSharedExpert = false;
    /// A flag indicating whether the shared experts have a dedicated gating mechanism.
    bool hasSharedExpertGate = false;
    /// The number of groups for grouped routing or parameter sharing.
    int numOfGroups = 0;
    /// The number of shared experts used in the model.
    int numOfSharedExperts = 0;
    /// The index of the first K dense layers to replace with sparse routing (-1 means no replacement).
    int firstKDenseReplace = 0;
    /// The identifier for the current layer in the model.
    int layerId = 0;
    /// The list of group IDs to which the top-K routing applies.
    atb::SVector<int32_t> topkGroups = {};
    /// The backend used for swiglu.
    atb_speed::common::OpBackend swigluBackend = atb_speed::common::OpBackend::ATB;

    // ==== Param for parallel strategy ====

    /// A flag indicating whether or not to use full tp in dense layer.
    bool isMlpFullTP = false;
    /// A flag indicating whether or not to use tensor parallelism in attention.
    bool hasAttnOprojTp = false;
    /// The rank of this device in the tensor parallelism communication domain in attention.
    int attnOprojTpRank = 0;
    /// The size of the tensor parallelism communication domain in attention.
    int attnOprojTpSize = 1;
    /// The communication domain of tensor parallelism in attention.
    std::string attnOprojTpDomain = "";
    /// The rankTableFile for the device in the attnTp communication domain.
    std::string attnOprojTpRankTableFile = "";
    std::string attnOprojTpBackend = "";
    bool attnOprojPrefetch = false;
    /// The rank of the current device within a lmhead TP communication domain
    int lmHeadTpRank = 0;
    /// The size of the lmhead TP communication domain that the current device is in
    int lmHeadTpSize = 1;
    /// The id of the lmhead TP communication domain
    std::string lmHeadTpDomain = "";
    /// A flag indicating whether the data collected after lmhead will again be data-wise separated
    bool lmHeadLocalTp = false;
    bool enableDpOut = false;
    /// A flag indicating whether the model utilizes expert parallelism.
    bool hasMoeEp = false;
    /// The rank of this device in the expert parallelism communication domain.
    int moeEpRank = 0;
    /// The size of the expert parallelism communication domain.
    int moeEpSize = 1;
    int maxDecodeDpTokenSize = 0;
    /// The communication domain of expert parallelism.
    std::string moeEpDomain = "";
    /// The rankTableFile for the device in the communication domain.
    std::string moeEpRankTableFile = "";
    std::string moeEpBackend = "";
    /// A flag indicating whether the model utilizes expert parallelism.
    bool hasMoeTp = false;
    /// The rank of this device in the expert parallelism communication domain.
    int moeTpRank = 0;
    /// The size of the expert parallelism communication domain.
    int moeTpSize = 1;
    /// The communication domain of expert parallelism.
    std::string moeTpDomain = "";
    /// The rankTableFile for the device in the communication domain.
    std::string moeTpRankTableFile = "";
    std::string moeTpBackend = "";
    /// The list of experts loaded on the device.
    std::vector<int32_t> deviceExpert = {};
    /// A vector that defines the quantization types of linears in MoE.
    /// It is automatically generated by MoE weight wrapper.
    std::vector<int> moeLinearQuantType = {};
    /// A vector that defines the quantization types of linears in MLP.
    /// It is automatically generated by MoE weight wrapper.
    std::vector<int> mlpLinearQuantType = {};
    /// A vector that defines the transpose types of linears in MoE.
    /// It is automatically generated by MoE weight wrapper.
    std::vector<int> moeLinearTransposeType = {};
    /// A vector that defines the transpose types of linears in MLP.
    /// It is automatically generated by MoE weight wrapper.
    std::vector<int> mlpLinearTransposeType = {};
    /// The number of experts selected for each token.
    atb::SVector<int32_t> numOfSelectedExperts = {2};

    // ==== Param for mix parallel data stream ====

    /// The number of stream in Attn module.
    int attnStreamNum = 1;
    /// The number of stream in FFN module.
    int ffnStreamNum = 1;
    /// The number of stream in Lmhead module.
    int lmheadStreamNum = 1;
    /// A flag indicating whether or not to allreduce in Attn module.
    bool attnAllreduce = false;
    /// A flag indicating whether or not to reducescatter in Attn module.
    bool attnReduceScatter = false;
    /// A flag indicating whether or not to reduce allgather in Attn module.
    bool attnAllGather = false;
    /// A flag indicating whether or not to reduce allreduce in FFN module.
    bool ffnAllreduce = false;
    /// A flag indicating whether or not to reducescatter in FFN module.
    bool ffnReduceScatter = false;
    /// A flag indicating whether or not to reduce allgather in FFN module.
    bool ffnAllGather = false;
    /// A flag indicating if Attn module has communication operation.
    bool hasAttnComm = false;
    /// A flag indicating if FFN module has communication operation.
    bool hasFfnComm = false;

    // ==== Param for moe ====

    /// A flag indicating whether to enable cumulative sum output for expert routing.
    bool enableExpertCumSumOutput = false;
    /// A flag indicating whether to implement dynamic ep.
    bool isDynamicEp = false;
    /// A flag indicating whether to implement load balance.
    bool enableLoadBalance = false;
    bool enableEPWB = false;
    bool enableLcocAll2All = false;
    bool enableATBGateMatmul = false;
    bool enableAllToAllMC2 = false;
    /// A flag indicating whether the model use cube and vector parallel
    bool enableCVOverlap = false;
    uint32_t numOfRedundantExpert = 0;
    int64_t numDanglingSharedExperts = 0;
    bool enableGatingDp = false;
    bool enableSharedExpertOverlap = false;
    /// The hccl comm of dispatch and combine.
    HcclComm dispatchAndCombineHcclComm;
    std::string dispatchAndCombinecommDomain = "";
};

/// A template class for representing a MoE Layer.
/// \tparam NormType atb_speed::base::RMS_NORM or atb::infer::LayerNormParam.
template <typename NormType>
class MoeDecoderLayer : public atb_speed::base::DecoderLayer<NormType> {
public:
    explicit MoeDecoderLayer(const MoeLayerParam &param);
    ~MoeDecoderLayer() override {};

    /// Create an graph operation that represents the structure of a layer
    /// \param operation the address of a pointer to a default operation
    /// \return A flag indicates whether the operation was successfully created.
    atb::Status BuildGraph(atb::Operation **operation) override;

protected:
    /// A function that constructs a map from in tensor names to in tensor ids.
    void ConstructInTensorMap() override;
    /// A function that constructs a map from internal tensor names to in tensor ids.
    void ConstructInternalTensorMap() override;
    /// A function that constructs a map from layer tensor names to InsertNorm function tensor names.
    std::map<std::string, uint32_t> ConstructNormTensorMap() const;
    /// Configure the parameters of the linear component within the fusion attention module
    /// \param fusionAttentionParam a reference to the funsion attention parameter to be set
    void SetFusionAttentionLinearParam(
        atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam) override;
    /// A function that update parameters of a `SparseMoeParam` based on parsed MoE layer parameters.
    /// \param sparseMoeParam An `atb_speed::common::SparseMoeParam` object that needs to be updated.
    virtual void SetSparseMoeParam(atb_speed::common::SparseMoeParam &sparseMoeParam);
    /// A function that update parameters of a `SparseMoeParam` based on parsed MoE layer parameters.
    /// \param sparseMoeParam An `atb_speed::common::SparseMoeParam` object that needs to be updated.
    virtual void SetSparseMoeCommParam(atb_speed::common::SparseMoeParam &sparseMoeParam);
    /// A function that add operations to a layer graph.
    /// \return A flag that indicates whether operations are added successfully.
    atb::Status AddOperationToGraph() override;
    /// A function that add attention node in a MoE layer.
    /// \return A flag that indicates whether the attention node is added successfully.
    virtual atb::Status AddAttention();
    /// A function that add a padding node for attention module's output.
    /// \return A flag that indicates whether the padding node is added successfully.
    virtual atb::Status AddAttnOutPadding();
    /// A function that add a padding node for residual add.
    /// \return A flag that indicates whether the padding node is added successfully.
    virtual atb::Status AddResidualPadding();
    /// A function that add a slice node for residual add.
    /// \return A flag that indicates whether the slice node is added successfully.
    virtual atb::Status AddResidualSliceNode();
    /// A function that add a reduce scatter node.
    /// \return A flag that indicates whether the reduce scatter node is added successfully.
    virtual atb::Status AddAttnReduceScatter();
    /// A function that update parameters of a `NormLinearParam` based on parsed MoE layer parameters.
    /// \param selfNormParam An `atb_speed::common::NormLinearParam<NormType>` object that needs to be updated.
    virtual void SetSelfNormParam(atb_speed::common::NormLinearParam<NormType> &selfNormParam);
    


    /// A function that add a add+norm fuse node in a MoE layer.
    /// \return A flag that indicates whether the residual node is added successfully.
    atb::Status SetAddNormNode();
    /// A function that add a residual add node in a MoE layer.
    /// \return A flag that indicates whether the residual node is added successfully.
    atb::Status AddFusionAttentionResidualAdd();
    /// A function that add a normalizaton node in a MoE layer.
    /// \return A flag that indicates whether the normalizaton node is added successfully.
    virtual atb::Status AddSelfNorm();
    /// A function that add a allgather node after attention module.
    /// \return A flag that indicates whether the allgather node is added successfully.
    virtual atb::Status AddAttnAllGather();
    /// A function that add a uppadding node after attention module.
    /// \return A flag that indicates whether the unpadding node is added successfully.
    virtual atb::Status AddAttnOutUnpadding();
    /// A function that add nodes in post attention process in a MoE layer.
    /// \return A flag that indicates whether the post attention process node is added successfully.
    virtual atb::Status AddPostAttentionProcess();
    /// A function that update parameters of a `NormLinearParam` based on parsed MoE layer parameters.
    /// \param SharedExpertParam An `atb_speed::common::SharedExpertParam` object that needs to be updated.
    virtual void SetSharedExpertParam(atb_speed::common::SharedExpertParam &sharedExpertParam);
    /// A function that add a MLP node in a MoE layer.
    /// \param SharedExpertParam An `atb_speed::common::SharedExpertParam` object for MLP node.
    /// \return A flag that indicates whether the MLP node is added successfully.
    virtual atb::Status AddMlpExpert(const atb_speed::common::SharedExpertParam &mlpExpertParam);
    /// A function that add a MoE node in a MoE layer.
    /// \return A flag that indicates whether the MoE node is added successfully.
    virtual atb::Status AddMoe();
    /// A function that add a Add node for shared experts and routing experts.
    /// \return A flag that indicates whether the Add node is added successfully.
    virtual atb::Status AddExpertAdd();
    /// A function that add a FFN node in a MoE layer.
    /// \return A flag that indicates whether the FFN node is added successfully.
    virtual atb::Status AddFFN();
    /// A function that add a padding node for FFN module's output.
    /// \return A flag that indicates whether the FFN output node is added successfully.
    virtual atb::Status AddFFNOutPadding();
    /// A function that add an allReduce node after a MoE node in a MoE layer.
    /// \return A flag that indicates whether the allReduce node is added successfully.
    virtual atb::Status AddMoeAllReduce();
    /// A function that add reduce scatter node after FFN module.
    /// \return A flag that indicates whether the reduce scatter node is added successfully.
    virtual atb::Status AddFFNReduceScatter();
    /// A function that add allgather node after FFN module.
    /// \return A flag that indicates whether the allgather node is added successfully.
    virtual atb::Status AddFFNAllGatherNode();
    /// A function that add a uppadding node after FFN module.
    /// \return A flag that indicates whether the unpadding node is added successfully.
    virtual atb::Status AddFFNOutUnPadding();
    /// A function that add nodes in post FFN process in a MoE layer.
    /// \return A flag that indicates whether the FFN node is added successfully.
    virtual atb::Status AddPostFFNProcess();
    /// Add the residual add node to the graph to conduct the add operation after the mlp node
    /// \return A flag indicates whether the operation was successfully added to the graph.
    atb::Status AddMlpResidualAdd();

    MoeLayerParam param;
    /// The index of the GATEUP linear within the moe
    const uint64_t moeGateupLinearIndex = 1;
    /// The index of the down linear within the moe
    const uint64_t moeDownLinearIndex = 3;
};
}  // namespace moe
}  // namespace atb_speed
#endif  // ATB_SPEED_MODELS_MOE_DECODER_LAYER_H
