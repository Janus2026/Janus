#include "fused_moe.h"

#include <glog/logging.h>

namespace janus {
namespace layer {

FusedMoEImpl::FusedMoEImpl(const ModelArgs& /*model_args*/,
                           const FusedMoEArgs& /*moe_args*/,
                           const QuantArgs& /*quant_args*/,
                           const ParallelArgs& /*parallel_args*/,
                           const torch::TensorOptions& /*options*/) {
  NOT_IMPLEMENTED_WITH_MSG(
      "FusedMoE is not supported for this backend. Please use MLU or ILU "
      "backend for MoE models.");
}

torch::Tensor FusedMoEImpl::forward_experts(
    const torch::Tensor& /*hidden_states*/,
    const torch::Tensor& /*router_logits*/,
    bool /*enable_all2all_communication*/) {
  NOT_IMPLEMENTED_WITH_MSG(
      "FusedMoE is not supported for this backend. Please use MLU or ILU "
      "backend for MoE models.");
  return torch::Tensor();
}

torch::Tensor FusedMoEImpl::forward(const torch::Tensor& /*hidden_states*/,
                                    const ModelInputParams& /*input_params*/) {
  NOT_IMPLEMENTED_WITH_MSG(
      "FusedMoE is not supported for this backend. Please use MLU or ILU "
      "backend for MoE models.");
  return torch::Tensor();
}

void FusedMoEImpl::load_state_dict(const StateDict& /*state_dict*/) {
  NOT_IMPLEMENTED_WITH_MSG(
      "FusedMoE is not supported for this backend. Please use MLU or ILU "
      "backend for MoE models.");
}

}  // namespace layer
}  // namespace janus
