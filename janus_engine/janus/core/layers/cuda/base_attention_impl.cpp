#include "base_attention_impl.h"

#include "kernels/cuda/utils.h"

namespace janus {
namespace layer {

BaseAttentionImpl::BaseAttentionImpl(int64_t num_heads,
                                     int64_t head_size,
                                     float scale,
                                     int64_t num_kv_heads,
                                     int64_t sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window) {
  // we only support bf16 kvcache for now
  decode_use_tensor_core_ = janus::kernel::cuda::should_use_tensor_core(
      /*kv_cache_dtype=*/torch::ScalarType::BFloat16, num_heads, num_kv_heads);
}

}  // namespace layer
}  // namespace janus
