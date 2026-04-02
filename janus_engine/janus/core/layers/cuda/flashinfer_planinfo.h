#pragma once

#include "layers/common/attention_metadata.h"

namespace janus::layer::flashinfer {

void update_prefill_plan_info(std::shared_ptr<PlanInfo> plan_info,
                              const std::string& backend,
                              const AttentionMetadata& attn_meta,
                              c10::ScalarType query_dtype,
                              c10::ScalarType key_dtype,
                              c10::ScalarType output_dtype,
                              int32_t head_dim_qk,
                              int32_t head_dim_vo,
                              int32_t num_qo_heads,
                              int32_t num_kv_heads,
                              bool enable_cuda_graph);

void update_chunked_prefill_plan_info(std::shared_ptr<PlanInfo> plan_info,
                                      const std::string& backend,
                                      const AttentionMetadata& attn_meta,
                                      c10::ScalarType query_dtype,
                                      c10::ScalarType key_dtype,
                                      c10::ScalarType output_dtype,
                                      int32_t head_dim_qk,
                                      int32_t head_dim_vo,
                                      int32_t num_qo_heads,
                                      int32_t num_kv_heads,
                                      int32_t block_size,
                                      int32_t window_size_left,
                                      bool enable_cuda_graph,
                                      bool causal = true);

void update_decode_plan_info(std::shared_ptr<PlanInfo> plan_info,
                             const std::string& backend,
                             const AttentionMetadata& attn_meta,
                             c10::ScalarType query_dtype,
                             c10::ScalarType key_dtype,
                             c10::ScalarType output_dtype,
                             int32_t head_dim_qk,
                             int32_t head_dim_vo,
                             int32_t num_qo_heads,
                             int32_t num_kv_heads,
                             int32_t block_size,
                             int32_t window_size_left,
                             bool enable_cuda_graph,
                             bool use_tensor_core);

}  // namespace janus::layer::flashinfer
