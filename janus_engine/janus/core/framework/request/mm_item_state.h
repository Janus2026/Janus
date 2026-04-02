#pragma once

#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

#include "core/util/hash_util.h"

namespace janus {

struct MMItemState {
  struct TokenPos {
    uint32_t offset = 0;
    uint32_t length = 0;
  };

  struct PrefixCache {
    XXH3Key key;
    uint32_t cached_token_num = 0;
  };

  const TokenPos& token_pos() const { return token_pos_; }
  TokenPos& mutable_token_pos() { return token_pos_; }

  const PrefixCache& prefix_cache() const { return prefix_cache_; }
  PrefixCache& mutable_prefix_cache() { return prefix_cache_; }

  bool prefix_cached() const;
  bool prefix_complete_cached() const;

 private:
  TokenPos token_pos_;
  PrefixCache prefix_cache_;
};

}  // namespace janus
