#include "lm_head_manual_loader.h"

namespace janus {
namespace layer {

LmHeadManualLoader::LmHeadManualLoader(uint64_t weight_count,
                                       const ModelContext& context)
    : BaseManualLoader(weight_count, context) {
  auto options = context.get_tensor_options();
  at_weight_tensors_[0] = torch::zeros({1}).to(options);
}

void LmHeadManualLoader::load_state_dict(const StateDict& state_dict) {
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

void LmHeadManualLoader::verify_loaded_weights(
    const std::string& weight_str) const {
  if (is_pinned_host_cache_hit()) {
    return;
  }
  CHECK(at_host_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "final lm_head weight is not loaded for " << weight_str;
}

void LmHeadManualLoader::merge_host_at_weights() {}

}  // namespace layer
}  // namespace janus
