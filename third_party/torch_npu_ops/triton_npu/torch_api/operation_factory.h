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

#include <memory>
#include <type_traits>
#include <unordered_map>

#include "operations.h"

namespace janus::kernel::npu {

class OperationFactory final {
 public:
  static OperationFactory& instance() {
    static OperationFactory inst;
    return inst;
  }

  RopeInplaceOp& rope_inplace() {
    // input param is key for OperationFactory
    return get_or_create<RopeInplaceOp>("rope_inplace_kernel");
  }

  FusedGdnGatingOp& fused_gdn_gating() {
    return get_or_create<FusedGdnGatingOp>("fused_gdn_gating_decode_kernel");
  }

  LayerNormFwdOp& layer_norm_fwd() {
    return get_or_create<LayerNormFwdOp>("layer_norm_fwd_kernel");
  }

  RecurrentGatedDeltaRuleFwdOp& recurrent_gated_delta_rule_fwd() {
    return get_or_create<RecurrentGatedDeltaRuleFwdOp>(
        "fused_recurrent_gated_delta_rule_fwd_kernel");
  }

  CausalConv1dUpdateNoCacheNoMtpOp& causal_conv1d_update_no_cache_no_mtp() {
    return get_or_create<CausalConv1dUpdateNoCacheNoMtpOp>(
        "_causal_conv1d_update_kernel_no_cache_len_no_mtp");
  }

  CausalConv1dUpdateQwenDecodeOp& causal_conv1d_update_qwen_decode() {
    return get_or_create<CausalConv1dUpdateQwenDecodeOp>(
        "_causal_conv1d_update_qwen_decode_kernel");
  }

 private:
  OperationFactory() = default;

  template <class T>
  T& get_or_create(const char* key) {
    static_assert(std::is_base_of_v<OperationBase, T>,
                  "T must derive from OperationBase");
    auto it = ops_.find(key);
    if (it != ops_.end()) {
      return *static_cast<T*>(it->second.get());
    }
    auto p = std::make_unique<T>();
    T* raw = p.get();
    ops_.emplace(key, std::move(p));
    return raw[0];
  }

  std::unordered_map<std::string, std::unique_ptr<OperationBase>> ops_;
};

}  // namespace janus::kernel::npu
