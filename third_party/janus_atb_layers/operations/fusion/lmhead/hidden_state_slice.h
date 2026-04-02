/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 */

#ifndef ATB_SPEED_MODELS_COMMON_LAYER_HIDDEN_STATE_SLICE_H
#define ATB_SPEED_MODELS_COMMON_LAYER_HIDDEN_STATE_SLICE_H

#include "atb/atb_infer.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear_parallel.h"

namespace atb_speed {
namespace common {
/// A struct defines `HiddenStateSlice` operation's parameters.
struct HiddenStateSliceParam {
    /// The rank of this device in the tensor parallelism communication domain in lmhead.
    int rank = 0;
    /// The size of the tensor parallelism communication domain in lmhead.
    int world_size = 1;
};

/// Create `HiddenStateSlice` graph operation.
/// \param param `HiddenStateSlice`'s parameters, see `HiddenStateSliceParam` for more details.
/// \param operation The address pointer to the `HiddenStateSlice` operation.
/// Operation's Inputs:
/// Name                   | Dtype | Shape |
/// ---------------------- | ----- | ----- |
/// in_hidden_states       | float16/float/int8/bool/int32/uint32/bf16 | [all_token_size, vocab_size] |
/// Operation's Outputs:
/// Name                   | Dtype | Shape |
/// ---------------------- | ----- | ----- |
/// output                 | float16/float/int8/bool/int32/uint32/bf16 | [token_size, vocab_size] |
atb::Status HiddenStateSlice(const HiddenStateSliceParam &param, atb::Operation **operation);
}  // namespace common
}  // namespace atb_speed
#endif
