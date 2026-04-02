#pragma once

#include <torch/torch.h>

#include <memory>
#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"

namespace janus {
namespace layer {

class BaseAttentionImpl;

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

 private:
  std::shared_ptr<BaseAttentionImpl> attention_impl_;
};
TORCH_MODULE(Attention);

}  // namespace layer
}  // namespace janus