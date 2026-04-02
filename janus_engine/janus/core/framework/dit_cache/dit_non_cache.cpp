#include "dit_non_cache.h"

namespace janus {

void DiTNonCache::init(const DiTCacheConfig& cfg) {
  // NonCache: nothing to initialize.
}

bool DiTNonCache::on_before_block(const CacheBlockIn& blockin) { return false; }

CacheBlockOut DiTNonCache::on_after_block(const CacheBlockIn& blockin) {
  TensorMap out_map;
  out_map["hidden_states"] =
      get_tensor_or_empty(blockin.tensors, "hidden_states");
  out_map["encoder_hidden_states"] =
      get_tensor_or_empty(blockin.tensors, "encoder_hidden_states");
  return CacheBlockOut(out_map);
}

bool DiTNonCache::on_before_step(const CacheStepIn& stepin) { return false; }

CacheStepOut DiTNonCache::on_after_step(const CacheStepIn& stepin) {
  TensorMap out;
  out["hidden_states"] = get_tensor_or_empty(stepin.tensors, "hidden_states");
  return CacheStepOut(out);
}

}  // namespace janus
