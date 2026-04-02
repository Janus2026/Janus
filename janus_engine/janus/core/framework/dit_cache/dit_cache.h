#pragma once
#include "dit_cache_impl.h"

namespace janus {

class DiTCache {
 public:
  DiTCache() = default;
  ~DiTCache() = default;

  DiTCache(const DiTCache&) = delete;
  DiTCache& operator=(const DiTCache&) = delete;
  DiTCache(DiTCache&&) = delete;
  DiTCache& operator=(DiTCache&&) = delete;

  static DiTCache& get_instance() {
    static DiTCache ditcache;
    return ditcache;
  }

  bool init(const DiTCacheConfig& cfg);

  bool on_before_block(const CacheBlockIn& blockin);
  CacheBlockOut on_after_block(const CacheBlockIn& blockin);

  bool on_before_step(const CacheStepIn& stepin);
  CacheStepOut on_after_step(const CacheStepIn& stepin);

 private:
  std::unique_ptr<DitCacheImpl> active_cache_;
};

}  // namespace janus
