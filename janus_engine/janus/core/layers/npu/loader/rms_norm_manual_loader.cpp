#include "rms_norm_manual_loader.h"

namespace janus {
namespace layer {

RMSNormManualLoader::RMSNormManualLoader(uint64_t weight_count,
                                         const ModelContext& context)
    : BaseManualLoader(weight_count, context) {
  auto options = context.get_tensor_options();
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  at_weight_tensors_[0] = torch::zeros({1}).to(options);
  at_host_weight_tensors_[0] = torch::zeros({1});
}

void RMSNormManualLoader::load_state_dict(const StateDict& state_dict) {
  if (is_pinned_host_cache_hit()) {
    return;
  }
  set_weight(state_dict, "weight", 0, true);
  at_host_weight_tensors_[0] = at_host_weight_tensors_[0].to(dtype_);
}

void RMSNormManualLoader::verify_loaded_weights(
    const std::string& weight_str) const {
  if (is_pinned_host_cache_hit()) {
    return;
  }
  CHECK(at_host_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "final norm weight is not loaded for " << weight_str;
}

void RMSNormManualLoader::merge_host_at_weights() {}

}  // namespace layer
}  // namespace janus
