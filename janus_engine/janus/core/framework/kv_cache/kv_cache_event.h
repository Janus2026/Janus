#pragma once
#include <unordered_set>

#include "util/hash_util.h"

namespace janus {

struct KvCacheEvent {
  std::unordered_set<XXH3Key, FixedStringKeyHash, FixedStringKeyEqual>
      stored_cache;
  std::unordered_set<XXH3Key, FixedStringKeyHash, FixedStringKeyEqual>
      removed_cache;

  void clear() {
    stored_cache.clear();
    removed_cache.clear();
  }
};

}  // namespace janus
