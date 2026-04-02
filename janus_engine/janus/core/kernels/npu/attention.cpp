#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"
namespace janus::kernel::npu {

void reshape_paged_cache(torch::Tensor& key,
                         std::optional<torch::Tensor>& value,
                         torch::Tensor& k_cache,
                         std::optional<torch::Tensor>& v_cache,
                         const torch::Tensor& slot_mapping) {
  atb::npu_reshape_and_cache(
      key, value.value(), k_cache, v_cache.value(), slot_mapping);
}

void batch_prefill(const torch::Tensor& query,
                   const torch::Tensor& key,
                   const torch::Tensor& value,
                   const torch::Tensor& mask,
                   const torch::Tensor& seq_len,
                   float scale,
                   torch::Tensor& output) {
  int64_t num_heads = query.size(-2);
  int64_t num_kv_heads = key.size(-2);
  atb::npu_flash_attention(
      query, key, value, mask, seq_len, scale, num_heads, num_kv_heads, output);
}

void batch_decode(const torch::Tensor& query,
                  const torch::Tensor& k_cache,
                  const torch::Tensor& v_cache,
                  float scale,
                  const torch::Tensor& block_table,
                  const torch::Tensor& seq_lens,
                  torch::Tensor& output) {
  int64_t head_size = query.size(-1);
  int64_t num_heads = query.size(-2);
  int64_t num_kv_heads = k_cache.size(-2);
  auto q = query.view({-1, num_heads, head_size});
  auto o = output.view({-1, num_heads, head_size});
  atb::npu_paged_attention(q,
                           k_cache,
                           v_cache,
                           num_kv_heads,
                           num_heads,
                           scale,
                           block_table,
                           seq_lens,
                           o);
}

void batch_decode_acl_graph(const torch::Tensor& query,
                            const torch::Tensor& k_cache,
                            const torch::Tensor& v_cache,
                            float scale,
                            const torch::Tensor& block_table,
                            const torch::Tensor& seq_lens,
                            const torch::Tensor& tiling_data,
                            torch::Tensor& output) {
  int64_t head_size = query.size(-1);
  int64_t num_heads = query.size(-2);
  int64_t num_kv_heads = k_cache.size(-2);
  auto q = query.view({-1, num_heads, head_size});
  auto o = output.view({-1, num_heads, head_size});
  atb::npu_custom_paged_attention(q,
                                  k_cache,
                                  v_cache,
                                  num_kv_heads,
                                  num_heads,
                                  scale,
                                  block_table,
                                  seq_lens,
                                  tiling_data,
                                  o);
}

}  // namespace janus::kernel::npu
