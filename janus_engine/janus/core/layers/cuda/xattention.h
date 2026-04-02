#pragma once

#include <torch/torch.h>

#include <tuple>

#include "base_attention_impl.h"
#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"

namespace janus {
namespace layer {

// XAttentionImpl inherits from BaseAttentionImpl and provides
// the specific implementation for xattention backend.
class XAttentionImpl : public BaseAttentionImpl {
 public:
  XAttentionImpl(int64_t num_heads,
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

  void decoder_forward(const AttentionMetadata& attn_metadata,
                       torch::Tensor& query,
                       torch::Tensor& key,
                       torch::Tensor& value,
                       torch::Tensor& output);
  void run_single_stage_decode(const AttentionMetadata& attn_metadata,
                               const torch::Tensor& key,
                               torch::Tensor& query,
                               torch::Tensor& output);

  void run_two_stage_decode(const AttentionMetadata& attn_metadata,
                            torch::Tensor& query,
                            torch::Tensor& output);
};

}  // namespace layer
}  // namespace janus
