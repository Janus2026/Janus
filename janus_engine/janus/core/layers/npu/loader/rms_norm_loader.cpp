#include "rms_norm_loader.h"

namespace janus {
namespace layer {

RMSNormLoader::RMSNormLoader(uint64_t weight_count, const ModelContext& context)
    : BaseLoader(weight_count, context) {
  auto options = context.get_tensor_options();
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  at_weight_tensors_[0] = torch::zeros({1}).to(options);
}

void RMSNormLoader::load_state_dict(const StateDict& state_dict) {
  set_weight(state_dict, "weight", 0);
  at_weight_tensors_[0] = at_weight_tensors_[0].to(dtype_);
}

void RMSNormLoader::verify_loaded_weights(const std::string& weight_str) const {
  CHECK(at_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "final norm weight is not loaded for " << weight_str;
}

}  // namespace layer
}  // namespace janus