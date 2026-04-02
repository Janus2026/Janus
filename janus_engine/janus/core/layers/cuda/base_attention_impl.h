#pragma once

#include <torch/torch.h>

#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"

namespace janus {
namespace layer {

// Base class for different attention implementations.
// This class contains common member variables and defines the common interface.
class BaseAttentionImpl {
 public:
  BaseAttentionImpl(int64_t num_heads,
                    int64_t head_size,
                    float scale,
                    int64_t num_kv_heads,
                    int64_t sliding_window);

  virtual ~BaseAttentionImpl() = default;

  // Pure virtual function that must be implemented by derived classes
  virtual std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      torch::Tensor& output,
      KVCache& kv_cache) = 0;

 protected:
  // Common member variables shared by all attention implementations
  int64_t num_heads_;
  int64_t head_size_;
  float scale_;
  int64_t num_kv_heads_;
  int64_t sliding_window_;
  bool decode_use_tensor_core_;
};

}  // namespace layer
}  // namespace janus
