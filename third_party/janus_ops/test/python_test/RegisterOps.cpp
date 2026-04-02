/* Copyright 2025 The Janus Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at


Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

void select_unshared_kv_impl_npu(
    const at::Tensor& beam_index,
    std::vector<at::Tensor> x_key_block,
    std::vector<at::Tensor> x_value_block,
    const at::Tensor& block_table,
    const at::Tensor& group_token_num,
    int64_t decode_step,
    int64_t beam_size,
    int64_t layer_num) {
  at::TensorList x_key_block_list = at::TensorList(x_key_block);
  at::TensorList x_value_block_list = at::TensorList(x_value_block);
  EXEC_NPU_CMD(aclnnSelectUnsharedKV,
               beam_index,
               block_table,
               x_key_block_list,
               x_value_block_list,
               group_token_num,
               decode_step,
               beam_size,
               layer_num,
               x_key_block_list,
               x_value_block_list);
  return ;
}

std::tuple<at::Tensor&, at::Tensor&> cache_unshared_kv_impl_npu(
    at::Tensor& x_key_block,
    at::Tensor& x_value_block,
    const at::Tensor& curr_key,
    const at::Tensor& curr_value,
    const at::Tensor& block_table,
    const at::Tensor& decode_step) {
  EXEC_NPU_CMD(aclnnCacheUnsharedKv,
               x_key_block,
               x_value_block,
               curr_key,
               curr_value,
               block_table,
               decode_step,
               x_key_block,
               x_value_block);
  return std::tuple<at::Tensor&, at::Tensor&>(x_key_block, x_value_block);
}

at::Tensor x_attention_impl_npu(const at::Tensor& query,
                                const at::Tensor& key_cache,
                                const at::Tensor& value_cache,
                                const at::Tensor& unshared_key,
                                const at::Tensor& unshared_value,
                                const c10::optional<at::Tensor>& shared_block_tables,
                                const c10::optional<at::Tensor>& unshared_block_tables,
                                const at::Tensor& actual_shared_kvlen,
                                const at::Tensor& decode_step) {
  at::Tensor attnOut = at::empty_like(query);
  
  EXEC_NPU_CMD(aclnnXAttention,
               query,
               key_cache,
               value_cache,
               unshared_key,
               unshared_value,
               unshared_block_tables,
               actual_shared_kvlen,
               decode_step,
               shared_block_tables,
               attnOut);

  return attnOut;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> beam_search_group_impl_npu(
                                const at::Tensor& log_probs,
                                const at::Tensor& top_tokens,
                                const at::Tensor& top_probs,
                                const at::Tensor& sequence,
                                int64_t current_step
                              ) {
  auto output_shape = log_probs.sizes();
  // Allocate outputs on the same device with appropriate dtypes
  at::Tensor out_token_ids = at::zeros(output_shape, top_tokens.options());
  at::Tensor out_token_index = at::zeros(output_shape, top_tokens.options());
  at::Tensor out_log_probs = at::zeros(output_shape, log_probs.options());
  at::Tensor out_beam_count_prefix_sums = at::zeros(output_shape, top_tokens.options());
  at::Tensor out_sequence = at::zeros(sequence.sizes(), top_tokens.options());
  EXEC_NPU_CMD(aclnnBeamSearchGroup,
               log_probs, top_tokens, top_probs, sequence,current_step,
               out_token_ids, out_token_index, out_log_probs, out_beam_count_prefix_sums, out_sequence);
  return std::make_tuple(out_token_ids, out_token_index, out_log_probs, out_beam_count_prefix_sums, out_sequence);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("select_unshared_kv", &select_unshared_kv_impl_npu, "select_unshared_kv");
  m.def("cache_unshared_kv", &cache_unshared_kv_impl_npu, "cache_unshared_kv");
  m.def("x_attention", &x_attention_impl_npu, "x_attention");
  m.def("beam_search_group", &beam_search_group_impl_npu, "beam_search_group");
}