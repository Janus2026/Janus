#include "activation.h"

#include "kernels/ops_api.h"
namespace janus {
namespace layer {

ActivationImpl::ActivationImpl(const std::string& act_mode, bool is_gated)
    : act_mode_(act_mode), is_gated_(is_gated) {}

void ActivationImpl::forward(torch::Tensor& input, torch::Tensor& output) {
  janus::kernel::ActivationParams activation_params;
  activation_params.input = input;
  activation_params.output = output;
  activation_params.act_mode = act_mode_;
  activation_params.is_gated = is_gated_;
  janus::kernel::active(activation_params);
  // Unified assignment: NPU returns new tensor, others modify in-place (no-op
  // assignment)
  output = activation_params.output;
}

}  // namespace layer
}  // namespace janus