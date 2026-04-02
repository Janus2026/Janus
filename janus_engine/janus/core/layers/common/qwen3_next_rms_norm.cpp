#include "qwen3_next_rms_norm.h"

#include <glog/logging.h>

namespace janus {
namespace layer {

Qwen3NextRMSNormImpl::Qwen3NextRMSNormImpl(int64_t dim,
                                           double eps,
                                           const torch::TensorOptions& options)
    : norm_dim_(dim), eps_(eps) {
  weight_ = register_parameter("weight", torch::empty({dim}, options), false);
}

torch::Tensor Qwen3NextRMSNormImpl::forward(torch::Tensor& input) {
  auto input_dtype = input.dtype();
  input = input.to(torch::kFloat32);

  // Calculate RMS
  auto variance = torch::mean(torch::pow(input, 2), -1, true);
  auto normalized = input * torch::rsqrt(variance + eps_);

  // Apply weight and convert back to original dtype
  return (normalized * (1.0f + weight_.to(torch::kFloat32))).to(input_dtype);
}

void Qwen3NextRMSNormImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
}

}  // namespace layer
}  // namespace janus
