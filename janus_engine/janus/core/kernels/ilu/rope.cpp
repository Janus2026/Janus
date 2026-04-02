#include "ilu_ops_api.h"
#include "utils.h"

namespace janus::kernel::ilu {

void apply_rope_pos_ids_cos_sin_cache(torch::Tensor& query,
                                      torch::Tensor& key,
                                      torch::Tensor& cos_sin_cache,
                                      torch::Tensor& positions,
                                      bool interleave) {
  const int64_t head_size = cos_sin_cache.size(-1);
  infer::janus_rotary_embedding(
      positions, query, key, head_size, cos_sin_cache, !interleave);
}

}  // namespace janus::kernel::ilu
