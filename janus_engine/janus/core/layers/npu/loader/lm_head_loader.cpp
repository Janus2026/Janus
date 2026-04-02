#include "lm_head_loader.h"

namespace janus {
namespace layer {

LmHeadLoader::LmHeadLoader(uint64_t weight_count, const ModelContext& context)
    : BaseLoader(weight_count, context) {
  auto options = context.get_tensor_options();
  at_weight_tensors_[0] = torch::zeros({1}).to(options);
}

void LmHeadLoader::load_state_dict(const StateDict& state_dict) {
  if (dp_size_ > 1) {
    set_weight(
        state_dict, "weight", 0, 1, dp_local_tp_rank_, dp_local_tp_size_);
  } else {
    set_weight(state_dict, "weight", 0, 1);
  }
}

void LmHeadLoader::verify_loaded_weights(const std::string& weight_str) const {
  CHECK(at_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "final lm_head weight is not loaded for " << weight_str;
}

}  // namespace layer
}  // namespace janus
