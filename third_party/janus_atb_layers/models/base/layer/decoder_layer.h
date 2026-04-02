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
#ifndef ATB_SPEED_MODELS_BASE_DECODER_LAYER_H
#define ATB_SPEED_MODELS_BASE_DECODER_LAYER_H

#include <vector>
#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/log.h"

#include "models/base/param/layer_param.h"
#include "operations/fusion/attention/fusion_attention.h"
#include "operations/fusion/mlp/mlp.h"

namespace atb_speed {
namespace base {

/// Base class to define the structure of a layer in a large language models, called by `DecoderModel` class.
/// \tparam NormType The type of normalization; refer to `NormType` for more details.
template <typename NormType>
class DecoderLayer {
public:
    explicit DecoderLayer(const LayerParam &param);
    virtual ~DecoderLayer() {};

    /// Create an graph operation that represents the structure of a layer
    /// \param operation the address of a pointer to a default operation
    /// \return A flag indicates whether the operation was successfully created.
    virtual atb::Status BuildGraph(atb::Operation **operation);

protected:
    /// Construct a in tensor list by selecting input tensors from all the avaliable input tensor candidates based on
    // the provided parameters `param`.
    virtual void ConstructInTensorMap();
    /// Construct a in tensor list by selecting input tensors from all the avaliable internal tensor candidates
    /// based on the provided parameters `param`.
    virtual void ConstructInternalTensorMap();
    /// Duplicate inTensorCandidates["default"], internalTensorCandidates["default"],internalTensorCandidates["attn_dp"]
    virtual void DuplicateAuxiliaryTensorMap();
    /// The main entrance to set the fusion attention module's parameters
    /// \param fusionAttentionParam a reference to the funsion attention parameter to be set
    virtual void SetFusionAttentionParam(atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam);
    /// Configure the parameters of the normalization component within the fusion attention module
    /// \param fusionAttentionParam a reference to the funsion attention parameter to be set
    virtual void SetFusionAttentionNormParam(atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam);
    /// Configure the parameters of the linear component within the fusion attention module
    /// \param fusionAttentionParam a reference to the funsion attention parameter to be set
    virtual void SetFusionAttentionLinearParam(atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam);
    /// Configure the parameters of ATB's self-attention component within the fusion attention module
    /// \param fusionAttentionParam a reference to the funsion attention parameter to be set
    virtual void SetFusionAttentionATBSelfAttentionParam(
        atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam);
    /// Configure the parameters of ATB's paged attention component within the fusion attention module
    /// \param fusionAttentionParam a reference to the funsion attention parameter to be set
    virtual void SetFusionAttentionATBPagedAttentionParam(
        atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam);
    /// Configure the parameters of AclNN's attention component within the fusion attention module
    /// (used in the decode phase)
    /// \param fusionAttentionParam a reference to the funsion attention parameter to be set
    virtual void SetFusionAttentionAclNNIncreAttentionParam(
        atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam);
    /// The main entrance to set the mlp module's parameters
    //// \param mlpParam a reference to the mlp parameter to be set
    virtual void SetMlpParam(atb_speed::common::MlpParam<NormType> &mlpParam);
    /// Configure the parameters of the normalization component within the mlp module
    //// \param mlpParam a reference to the mlp parameter to be set
    virtual void SetMlpNormParam(atb_speed::common::MlpParam<NormType> &mlpParam);
    /// Create the fusion attention operation
    /// \param op the address of a pointer to a default operation
    atb::Status CreateFusionAttentionOperation(atb::Operation **op);
    /// Create the mlp operation
    /// \param op the address of a pointer to a default operation
    atb::Status CreateMlpOperation(atb::Operation **op);
    /// Get the fusion attention module's input tensor
    /// \return A map of the attention module's input tensor categories and their corresponding tensor names
    virtual std::map<unsigned int, std::vector<std::string>> GetAttentionIntensor();
    /// Get the mlp module's input tensor
    /// \return A map of the mlp module's input tensor categories and their corresponding tensor names
    virtual std::map<unsigned int, std::vector<std::string>> GetMlpIntensor();
    /// Update internal tensor candidates
    virtual void SetDefaultInternalTensorCandidates();
    /// The primary entry point for adding all operations to the graph in sequential order.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddOperationToGraph();
    /// Add the fusion attention node to the graph
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddFusionAttention(bool is_auxiliary, uint64_t stream_id);
    /// Add the residual add node to the graph to conduct the add operation after the fusion attention node
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddFusionAttentionResidualAdd(bool is_auxiliary, uint64_t stream_id);
    virtual atb::Status AddPostSelfAttnLayerNorm();
    virtual atb::Status AddPostMlpLayerNorm();
    /// Add the mlp node to the graph
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddMlp(bool is_auxiliary, uint64_t stream_id);
    /// Add the residual add node to the graph to conduct the add operation after the mlp node
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddMlpResidualAdd(bool is_auxiliary, uint64_t stream_id);
    /// Add the fused all gather node to the graph to gather information across deivces
    /// in the same communication domain. This function calls AddPadNode(), AddAllGather(), and AddUnPadNode().
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddFusedAllGather(bool is_auxiliary, uint64_t stream_id);
    /// Add the revert all gather node to the graph to drop unnecessary information before the attention operation
    /// of the next layer.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddRevertAllGather(bool is_auxiliary, uint64_t stream_id);
    /// Add the pad node to the graph to make sure that the lengths of the input tensor to AllGather operator
    /// across devices of the same communication domain are identical.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddPadNode(bool is_auxiliary, uint64_t stream_id);
    /// Add the all gather node to the graph to gather information across devices
    /// in the same communication domain.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddAllGatherNode(bool is_auxiliary, uint64_t stream_id);
    /// Add the unpad node to the graph to drop the excessive information padded before AllGather operator.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddUnPadNode(bool is_auxiliary, uint64_t stream_id);

    /// Specifies all potential input tensors, where the key represents the feature name,
    /// and the value corresponds to the input tensor name.
    std::map<std::string, std::vector<std::string>> inTensorCandidates = {};
    /// Specifies all potential internal tensors, where the key represents the feature name,
    /// and the value corresponds to the internal tensor name.
    std::map<std::string, std::vector<std::string>> internalTensorCandidates = {};
    /// A vector contains names of all the required input tensors.
    std::vector<std::string> inTensorList = {};
    /// A vector contains names of all the required intermediate tensors.
    std::vector<std::string> intermediateTensorList = {};
    /// A vector contains names of all the required output tensors.
    std::vector<std::string> outTensorList = {"out"};
    /// Defines all the required tensors for the current graph, with the key representing the input tensor name
    /// and the value corresponding to the tensor index.
    /// Tensors are ordered by input tensors, output tensors and internal tensors.
    std::map<std::string, uint32_t> tensorMap = {};

    /// Layer parameters
    LayerParam param;
    /// A layer graph to be created
    atb::GraphParam graph;

private:
    /// Default weight names required by the fusion attention node
    const std::vector<std::string> attnWeight = {
        // Pack:
        // MHA [3 * numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
        // GQA [(numAttentionHeadsPerRank + 2 * numKeyValueHeadsPerRank) * hiddenSizePerAttentionHead, hiddenSize]
        // No pack:
        // (Q) shape: [numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
        "in_qkv_weight_0", "in_qkv_bias_0", "in_qkv_descale_0", "in_qkv_offset_0", "in_qkv_scale_0",
        "in_qkv_compress_idx_0",
        // Pack: no usage; No pack: (K) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
        "in_qkv_weight_1", "in_qkv_bias_1", "in_qkv_descale_1", "in_qkv_offset_1", "in_qkv_scale_1",
        "in_qkv_compress_idx_1",
        // Pack: no usage; No pack: (V) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
        "in_qkv_weight_2", "in_qkv_bias_2", "in_qkv_descale_2", "in_qkv_offset_2", "in_qkv_scale_2",
        "in_qkv_compress_idx_2",
        // shape: [hiddenSize, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
        "in_qkv_dense_weight", "in_qkv_dense_bias", "in_qkv_dense_descale", "in_qkv_dense_offset",
        "in_qkv_dense_scale", "in_qkv_dense_compress_idx"};

    /// Default weight names required by the mlp attention node
    const std::vector<std::string> mlpWeight = {
        // Pack: shape: [2 * intermediateSizePerRank, hiddenSize]
        // No pack: (Gate) shape: [intermediateSizePerRank, hiddenSize]
        "in_mlp_weight_0", "in_mlp_bias_0", "in_mlp_descale_0", "in_mlp_offset_0", "in_mlp_scale_0",
        "in_mlp_compress_idx_0",
        // Pack: no usage; No pack: (Up) shape: [intermediateSizePerRank, hiddenSize]
        "in_mlp_weight_1", "in_mlp_bias_1", "in_mlp_descale_1", "in_mlp_offset_1", "in_mlp_scale_1",
        "in_mlp_compress_idx_1",
        // shape: [hiddenSize, intermediateSizePerRank]
        "in_mlp_down_weight", "in_mlp_down_bias", "in_mlp_down_descale", "in_mlp_down_offset",
        "in_mlp_down_scale", "in_mlp_down_compress_idx"};
};

}  // namespace base
}  // namespace atb_speed
#endif
