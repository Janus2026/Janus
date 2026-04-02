#include "mm_item_state.h"

namespace janus {

bool MMItemState::prefix_cached() const {
  return prefix_cache_.cached_token_num > 0;
}

bool MMItemState::prefix_complete_cached() const {
  if (!prefix_cached()) {
    return false;
  }

  return prefix_cache_.cached_token_num == token_pos_.length;
}

}  // namespace janus
