#include "rms_norm_gated.h"

#include <glog/logging.h>

#include "framework/state_dict/utils.h"
#include "janus/core/kernels/ops_api.h"

namespace janus {
namespace layer {

RmsNormGatedImpl::RmsNormGatedImpl(int64_t dim,
                                   double eps,
                                   const torch::TensorOptions& options)
    : norm_dim_(dim), eps_(eps) {
  weight_ = register_parameter(
      "weight", torch::empty({dim}, options), /*requires_grad=*/false);
}

torch::Tensor RmsNormGatedImpl::forward(torch::Tensor& input,
                                        std::optional<torch::Tensor> gate) {
  janus::kernel::GatedLayerNormParams params;
  auto input_type = input.dtype();
  input = input.to(torch::kFloat32);
  params.x = input;
  params.weight = weight_.to(torch::kFloat32);
  torch::Tensor bias;
  params.bias = bias;
  params.eps = eps_;
  if (gate.has_value()) {
    gate = gate.value().to(torch::kFloat32);
    params.z = gate;
  }
  params.group_size = input.size(-1);
  params.is_rms_norm = true;
  auto ret = janus::kernel::gated_layer_norm(params);
  return ret.to(input_type);
}

void RmsNormGatedImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
}

}  // namespace layer
}  // namespace janus
