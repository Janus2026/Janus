#include "attention.h"

#include "base_attention_impl.h"
#include "core/common/rec_model_utils.h"
#include "flashinfer_attention.h"
#include "xattention.h"

namespace janus {
namespace layer {
AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window) {
  // Select implementation based on mode. Use polymorphism via base class
  // pointer to manage different implementations.

  if (is_rec_multi_round_mode()) {
    attention_impl_ = std::make_shared<XAttentionImpl>(
        num_heads, head_size, scale, num_kv_heads, sliding_window);
  } else {
    attention_impl_ = std::make_shared<FlashInferAttentionImpl>(
        num_heads, head_size, scale, num_kv_heads, sliding_window);
  }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  // Create output tensor internally to unify the interface with other devices
  torch::Tensor output = torch::empty_like(query);

  // Use polymorphism to dispatch to the appropriate implementation,
  // making the code elegant and type-safe.
  return attention_impl_->forward(
      attn_metadata, query, key, value, output, kv_cache);
}

}  // namespace layer
}  // namespace janus