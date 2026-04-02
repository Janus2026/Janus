#pragma once

#include "custom_functions_npu/atb_common.h"

using namespace std;

namespace atb {

void npu_paged_attention(const at::Tensor& query,
                         const at::Tensor& key_cache,
                         const at::Tensor& value_cache,
                         int64_t num_kv_heads,
                         int64_t num_heads,
                         double scale_value,
                         const at::Tensor& block_table,
                         const at::Tensor& context_lens,
                         at::Tensor& out);

// Custom paged attention for ACL graph execution
// This variant avoids .to(kCPU) operations that break ACL graph capture
void npu_custom_paged_attention(const at::Tensor& query,
                                const at::Tensor& key_cache,
                                const at::Tensor& value_cache,
                                int64_t num_kv_heads,
                                int64_t num_heads,
                                double scale_value,
                                const at::Tensor& block_table,
                                const at::Tensor& context_lens,
                                const at::Tensor& tiling_data,
                                at::Tensor& out);

void npu_reshape_and_cache(const at::Tensor& key,
                           const at::Tensor& value,
                           at::Tensor& key_cache,
                           at::Tensor& value_cache,
                           const at::Tensor& slot_indices);

void npu_flash_attention(const at::Tensor& query,
                         const at::Tensor& key,
                         const at::Tensor& value,
                         const at::Tensor& mask,
                         const at::Tensor& seq_len,
                         const double scale_value,
                         const int64_t num_heads,
                         const int64_t num_kv_heads,
                         at::Tensor& out);

void npu_rotary_embedding(const at::Tensor& positions,
                          at::Tensor& query,
                          at::Tensor& key,
                          int64_t head_size,
                          const at::Tensor& cos_sin_cache,
                          bool is_neox_style);

}  // namespace atb
