#include "word_embedding_manual_loader.h"

namespace janus {
namespace layer {

WordEmbeddingManualLoader::WordEmbeddingManualLoader(
    uint64_t weight_count,
    const ModelContext& context)
    : BaseManualLoader(weight_count, context) {}

void WordEmbeddingManualLoader::load_state_dict(const StateDict& state_dict) {
  if (is_pinned_host_cache_hit()) {
    return;
  }
  if (dp_size_ > 1) {
    set_weight(
        state_dict, "weight", 0, 1, dp_local_tp_rank_, dp_local_tp_size_, true);
  } else {
    set_weight(state_dict, "weight", 0, 1, true);
  }
}

void WordEmbeddingManualLoader::verify_loaded_weights(
    const std::string& weight_str) const {
  if (is_pinned_host_cache_hit()) {
    return;
  }
  CHECK(at_host_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "weight is not loaded for " << weight_str;
}

void WordEmbeddingManualLoader::merge_host_at_weights() {}

}  // namespace layer
}  // namespace janus
