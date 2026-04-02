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
#ifndef ATB_SPEED_MODELS_COMMON_LMHEAD_H
#define ATB_SPEED_MODELS_COMMON_LMHEAD_H

#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"

namespace atb_speed {
namespace common {
/// A struct of `LmHead`'s parameters.
struct LmHeadParam {
    /// Whether to use gatherAhead. We recommand to use gatherAhead at prefill stage, only transform tokens needed
    /// into logits based on input IN_INDICES, so that reducing memory usage.
    /// gatherAhead is not recommended at decode stage for it will cost more time to call gather.
    bool gatherAhead = false;
    /// Whether input tensor is unpadded.
    /// If false, input tensor shape is [batch_size, seq_len]. For request shortter than seq_len, it will be padded.
    /// If true, input tensor shape is [(seq_len_1 + seq_len_2 + ... + seq_len_n)].
    bool unpadInputs = false;
    /// Hidden size of each attention head in row parallel setting.
    /// Only positive if linearParallelParam.parallelType is ROW_PARALLEL.
    int hiddenSizePerAttentionHead = 0;
    bool enableDpOut = false;
    /// Parameters passed to `LinearParallel`, see `operations/fusion/linear/linear_parallel.h` for more details.
    atb_speed::common::LinearParallelParam linearParallelParam;
};

/// Create an `LmHead` operation.
///
/// \param param `LmHead`'s parameters, see `LmHeadParam` for more details.
/// \param operation The address to be filled with the created operation object.
/// \return A flag indicating whether the operation is created successfully.
///
/// Operation's Inputs:
/// Name          | Dtype   | Shape | Description |
/// --------------|---------| ----- | -------- |
/// hidden_states | float16 or bfloat16 | [len(all seq_len), hidden_size] if unpadInputs is true, otherwise [bsz, seq_len, hidden_size] | / |
/// weight        | float16 or bfloat16 | Let origin weight shape be [vocab_size, hidden_size], if linearParallelParam.parallelType is COLUMN_PARALLEL, then [vocab_size / world_size, hidden_size], otherwise [vocab_size, hidden_size / world_size] | / |
/// scale         | float16 | [1]   | Place holder, not used |
/// offset        | float16 | [1]   | Place holder, not used |
/// descale       | float16 | [1]   | Place holder, not used |
/// bias          | float16 | [1]   | Place holder, not used |
/// indices       | int64   | int64 | Optional, only needed when gatherAhead is true |
///
/// Operation's Outputs:
/// Name          | Dtype | Shape |
/// --------------|-------| ----- |
/// logits        | float16 or bfloat16 | [len(all seq_len), vocab_size] if unpadInputs is true, otherwise [bsz, seq_len, vocab_size] |
///
/// Example:
/// \code
/// enum TensorIdx: uint32_t {
///     IN_HIDDEN_STATES_ID = 0,
///     IN_WEIGHT_ID,
///     IN_INDICES_ID,
///     OUT_LOGITS_ID,
///     PLACE_HOLDER_ID,
/// };
/// std::vector<atb::Tensor> Tensors = {...};   // Prepare tensors here.
/// atb::Operation *op = nullptr;
/// atb_speed::Model::Node lmHeadNode;
/// atb_speed::common::LmHeadParam lmHeadParam;
/// // Modify LmHeadParam's attributes if needed.
/// CHECK_OPERATION_STATUS_RETURN(LmHead(lmHeadParam, &op));
/// lmHeadNode.operation.reset(op);
/// // Assume the input and output tensors are already set in graph.
/// lmHeadNode.inTensors = {
///     Tensors.at(IN_HIDDEN_STATES_ID),
///     Tensors.at(IN_WEIGHT_ID),
///     Tensors.at(PLACE_HOLDER_ID),
///     Tensors.at(PLACE_HOLDER_ID),
///     Tensors.at(PLACE_HOLDER_ID),
///     Tensors.at(PLACE_HOLDER_ID),
///     Tensors.at(IN_INDICES_ID)
/// };
/// lmHeadNode.outTensors = {
///     Tensors.at(OUT_LOGITS_ID)
/// }
/// graph.nodes.push_back(lmHeadNode);  // Add operation to its graph.
/// \endcode
atb::Status LmHead(const LmHeadParam &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif
