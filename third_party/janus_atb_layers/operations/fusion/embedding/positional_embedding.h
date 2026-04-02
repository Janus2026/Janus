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
#ifndef ATB_SPEED_MODELS_COMMON_LAYER_POSITIONAL_EMBEDDING_H
#define ATB_SPEED_MODELS_COMMON_LAYER_POSITIONAL_EMBEDDING_H

#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "operations/fusion/linear/linear_parallel.h"

namespace atb_speed {
namespace common {
/// An enum to represent the type of rotary position embedding.
enum RotaryType : uint32_t {
    /// No rotary position embedding.
    NO_ROTARY = 0,
    /// 1D rotary position embedding.
    HALF_ROTARY,
    /// 2D rotary position embedding.
    ALL_ROTARY,
};

/// A struct defines `positional embedding`'s parameters.
struct RotaryPositionEmbeddingParam {
    atb_speed::common::RotaryType rotaryType = ALL_ROTARY;  /// The type of rotary position embedding.
    bool isFA = true;   /// A flag to indicate whether to use the flash attention.
    /// Number of attention heads per rank, which equals to `num_attention_heads` // `world_size`.
    /// `num_attention_heads` is defined in model_path -> config.json.
    int headNum = 0;
    /// Hidden size of each attention head, which equals to `hidden_size` / `num_attention_heads`.
    /// `hidden_size` and `num_attention_heads` is defined in model_path -> config.json.
    int headDim = 0;
    /// Number of key-value heads per rank, which equals to `num_key_value_heads` // `world_size`.
    /// `num_key_value_heads` is defined in model_path -> config.json if defined, otherwise `num_attention_heads`.
    int kvHeadNum = 0;
    atb::infer::RopeParam ropeParam;    /// Parameters to be passed through to ATB rope operation.
};

/// Create a `RotaryPositionEmbedding` operation.
///
/// This function supports 1d and 2d Rotary Position Embedding (RoPE). RoPE is a positional encoding technique that
/// encodes the relative position of tokens in a sequence by rotating the embedding vectors of the tokens.
/// For more details, check out the Rope paper: `RoFormer: Enhanced Transformer with Rotary Position Embedding`.
///
/// \param param Parameters of ROPE operation, see `RotaryPositionEmbeddingParam` for more details.
/// \param operation The address to be filled with the created operation object.
/// \return A flag indicating the status of the operation creation.
///
/// Operation's Inputs:
/// Name                | Dtype | Shape |
/// ------------------- | ----- | ----- |
/// query               | float16/bfloat16 | [len(all_seq), num_heads, head_dim] in PA case, [bsz, seq_len, num_heads, head_dim] in FA case |
/// key                 | float16/bfloat16 | same as query's shape |
/// rope_cos            | float16/float32/bfloat16, float32 to enable high-precision rope | [len(all_seq), head_dim/2] if and only if ropeParm.rotaryCoeff equals 2, else [len(all_seq), head_dim] |
/// rope_sin            | float16/float32/bfloat16, float32 to enable high-precision rope | [len(all_seq), head_dim/2] if and only if ropeParm.rotaryCoeff equals 2, else [len(all_seq), head_dim] |
/// seq_len             | int32/uint32 | [bsz] |
///
/// Operation's Outputs:
/// Name                | Dtype | Shape |
/// embedded_query      | same as input query | same as input query |
/// embedded_key        | same as input key | same as input key |
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_QUERY_ID = 0,
///     IN_KEY_ID,
///     IN_ROPE_COS_ID,
///     IN_ROPE_SIN_ID,
///     IN_SEQ_LEN_ID,
///     OUT_EMBEDDED_QUERY_ID,
///     OUT_EMBEDDED_KEY_ID
/// };
/// }
/// atb::Node ropeNode;
/// atb::speed::common::RotaryPositionEmbeddingParam ropeParam;
/// // Modify ropeParam's attributes if needed.
/// RotaryPositionEmbedding(ropeParam, &ropeNode.operation);
/// ropeNode.inTensorIds = {
///     IN_QUERY_ID,
///     IN_KEY_ID,
///     IN_ROPE_COS_ID,
///     IN_ROPE_SIN_ID
///     IN_SEQ_LEN_ID
/// };
/// ropeNode.outTensorIds = {
///     OUT_EMBEDDED_QUERY_ID,
///     OUT_EMBEDDED_KEY_ID
/// };
/// graph.nodes.push_back(ropeNode);  // Add node to its graph.
/// \endcode
atb::Status RotaryPositionEmbedding(const RotaryPositionEmbeddingParam &param, atb::Operation **operation);

/// Create a `PositionalEmbeddingGather` operation.
///
/// This function get the positional embedding from the cosine table and sine table according to the position index.
///
/// \param operation The address to be filled with the created operation object.
///
/// Operation's Inputs:
/// Name               | Dtype | Shape |
/// -------------------|-------|-------|
/// position_ids       | int64/int32/uint32 | [len(all_seq)] or [bsz, seq_len] |
/// cosine_table       | float16/bfloat16 | depends on the model |
/// sine_table         | float16/bfloat16 | depends on the model |
///
/// Operation's Outputs:
/// Name               | Dtype | Shape |
/// -------------------|-------|-------|
/// cos_embedding      | same as cosine_table | [len(all_seq), cosine_table.shape[:]] or [bsz, seq_len, cosine_table.shape[:]] |
/// sin_embedding      | same as sine_table | same as cos_embedding |
///
/// Example:
/// \code
/// enum TensorIdx: uint32_t {
///     IN_POSITION_IDS_ID = 0,
///     IN_COSINE_TABLE_ID,
///     IN_SINE_TABLE_ID,
///     OUT_COSINE_EMBEDDING_ID,
///     OUT_SINE_EMBEDDING_ID
/// };
/// std::vector<atb::Tensor> Tensors = {...};   // Prepare tensors here.
/// atb::Operation *op = nullptr;
/// atb_speed::Model::Node positionalEmbeddingGatherNode;
/// CHECK_OPERATION_STATUS_RETURN(atb_speed::common::PositionalEmbeddingGather(&op));
/// positionalEmbeddingGatherNode.operation.reset(op);
/// // Assume the input and output tensors are already set in graph.
/// positionalEmbeddingGatherNode.inTensors = {
///     Tensors.at(IN_POSITION_IDS_ID),
///     Tensors.at(IN_COSINE_TABLE_ID,
///     Tensors.at(IN_SINE_TABLE_ID)
/// };
/// positionalEmbeddingGatherNode.outTensors = {
///     Tensors.at(OUT_COS_EMBEDDING_ID),
///     Tensors.at(OUT_SIN_EMBEDDING_ID)
/// };
/// graph.nodes.push_back(postionalEmbeddingGatherNode);  // Add node to its graph.
/// \endcode
atb::Status PositionalEmbeddingGatherV2(atb::Operation** operation);
atb::Status PositionalEmbeddingGather(atb::Operation **operation);
}  // namespace common
}  // namespace atb_speed
#endif
