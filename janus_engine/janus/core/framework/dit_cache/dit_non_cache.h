#pragma once
#include "dit_cache_impl.h"

namespace janus {

class DiTNonCache : public DitCacheImpl {
 public:
  DiTNonCache() = default;
  ~DiTNonCache() override = default;

  DiTNonCache(const DiTNonCache&) = delete;
  DiTNonCache& operator=(const DiTNonCache&) = delete;
  DiTNonCache(DiTNonCache&&) = default;
  DiTNonCache& operator=(DiTNonCache&&) = default;

  void init(const DiTCacheConfig& cfg) override;

  bool on_before_block(const CacheBlockIn& blockin) override;
  CacheBlockOut on_after_block(const CacheBlockIn& blockin) override;

  bool on_before_step(const CacheStepIn& stepin) override;
  CacheStepOut on_after_step(const CacheStepIn& stepin) override;
};

}  // namespace janus
