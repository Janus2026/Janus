/* Copyright 2026 The Janus Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#pragma once

#include <string>

#include "operation_base.h"

namespace janus::kernel::npu {

class RopeInplaceOp final : public OperationBase {
 public:
  RopeInplaceOp() : OperationBase("rope_inplace_kernel") {}
};

class FusedGdnGatingOp final : public OperationBase {
 public:
  FusedGdnGatingOp() : OperationBase("fused_gdn_gating_decode_kernel") {}
};

class LayerNormFwdOp final : public OperationBase {
 public:
  LayerNormFwdOp() : OperationBase("layer_norm_fwd_kernel") {}
};

class RecurrentGatedDeltaRuleFwdOp final : public OperationBase {
 public:
  RecurrentGatedDeltaRuleFwdOp()
      : OperationBase("fused_recurrent_gated_delta_rule_fwd_kernel") {}
};

class CausalConv1dUpdateNoCacheNoMtpOp final : public OperationBase {
 public:
  CausalConv1dUpdateNoCacheNoMtpOp()
      : OperationBase("_causal_conv1d_update_kernel_no_cache_len_no_mtp") {}
};

class CausalConv1dUpdateQwenDecodeOp final : public OperationBase {
 public:
  CausalConv1dUpdateQwenDecodeOp()
      : OperationBase("_causal_conv1d_update_qwen_decode_kernel") {}
};

}  // namespace janus::kernel::npu
