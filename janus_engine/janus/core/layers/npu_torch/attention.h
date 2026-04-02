#pragma once

#include <torch/torch.h>

#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "layers/common/attention_metadata.h"

namespace janus {
namespace layer {

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl() = default;

  AttentionImpl(int64_t num_heads,
                int64_t head_size,
                float scale,
                int64_t num_kv_heads,
                int64_t sliding_window);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      KVCache& kv_cache);

  void prefill_forward(torch::Tensor& query,
                       torch::Tensor& key,
                       torch::Tensor& value,
                       torch::Tensor& output,
                       const torch::Tensor& k_cache,
                       const std::optional<torch::Tensor>& v_cache,
                       const AttentionMetadata& attn_metadata);

  void decoder_forward(torch::Tensor& query,
                       torch::Tensor& output,
                       const torch::Tensor& k_cache,
                       const std::optional<torch::Tensor>& v_cache,
                       const AttentionMetadata& attn_metadata);

 private:
  int64_t num_heads_;
  int64_t head_size_;
  float scale_;
  int64_t num_kv_heads_;
  int64_t sliding_window_;
};
TORCH_MODULE(Attention);

}  // namespace layer
}  // namespace janus
