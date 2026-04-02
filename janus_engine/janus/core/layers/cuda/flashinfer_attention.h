#pragma once

#include <torch/torch.h>

#include <tuple>

#include "base_attention_impl.h"
#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"

namespace janus {
namespace layer {

// FlashInferAttentionImpl implements the standard attention computation using
// flashinfer kernels. This is the default implementation used when not in
// pure device mode.
class FlashInferAttentionImpl : public BaseAttentionImpl {
 public:
  FlashInferAttentionImpl(int64_t num_heads,
                          int64_t head_size,
                          float scale,
                          int64_t num_kv_heads,
                          int64_t sliding_window);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      torch::Tensor& output,
      KVCache& kv_cache) override;

 private:
  void prefill_forward(const AttentionMetadata& attn_metadata,
                       torch::Tensor& query,
                       torch::Tensor& key,
                       torch::Tensor& value,
                       torch::Tensor& output,
                       std::optional<at::Tensor>& output_lse);

  void chunked_prefill_forward(const AttentionMetadata& attn_metadata,
                               torch::Tensor& query,
                               const torch::Tensor& key,
                               torch::Tensor& output,
                               std::optional<at::Tensor>& output_lse,
                               const torch::Tensor& k_cache,
                               const torch::Tensor& v_cache);

  void decoder_forward(const AttentionMetadata& attn_metadata,
                       torch::Tensor& query,
                       const torch::Tensor& key,
                       torch::Tensor& output,
                       std::optional<at::Tensor>& output_lse,
                       const torch::Tensor& k_cache,
                       const torch::Tensor& v_cache);

 private:
  torch::Tensor float_workspace_buffer_;
  torch::Tensor int_workspace_buffer_;
  torch::Tensor page_locked_int_workspace_buffer_;
};

}  // namespace layer
}  // namespace janus
