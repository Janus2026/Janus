#pragma once

#include "layers/common/attention_metadata.h"

namespace janus::layer::xattention {

void update_xattention_plan_info(std::shared_ptr<PlanInfo> plan_info,
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
                                 bool causal,
                                 bool use_tensor_core,
                                 bool is_shared_stage_plan);

}  // namespace janus::layer::xattention
