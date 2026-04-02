#include "dit_cache.h"

namespace janus {

bool DiTCache::init(const DiTCacheConfig& cfg) {
  active_cache_ = create_dit_cache(cfg);
  if (!active_cache_) {
    return false;
  }
  active_cache_->init(cfg);
  return true;
}

bool DiTCache::on_before_block(const CacheBlockIn& blockin) {
  return active_cache_->on_before_block(blockin);
}

CacheBlockOut DiTCache::on_after_block(const CacheBlockIn& blockin) {
  return active_cache_->on_after_block(blockin);
}

bool DiTCache::on_before_step(const CacheStepIn& stepin) {
  return active_cache_->on_before_step(stepin);
}

CacheStepOut DiTCache::on_after_step(const CacheStepIn& stepin) {
  return active_cache_->on_after_step(stepin);
}

}  // namespace janus
