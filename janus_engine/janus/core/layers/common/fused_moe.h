#pragma once

#include <torch/torch.h>

#include "dense_mlp.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "fused_moe_base.h"
#include "linear.h"

namespace janus {
namespace layer {

// FusedMoE common implementation - placeholder for unsupported backends
// Actual implementation is in layers/mlu/fused_moe.h and layers/ilu/fused_moe.h
class FusedMoEImpl : public torch::nn::Module {
 public:
  FusedMoEImpl() = default;
  FusedMoEImpl(const ModelArgs& model_args,
               const FusedMoEArgs& moe_args,
               const QuantArgs& quant_args,
               const ParallelArgs& parallel_args,
               const torch::TensorOptions& options);

  torch::Tensor forward_experts(const torch::Tensor& hidden_states,
                                const torch::Tensor& router_logits,
                                bool enable_all2all_communication);
  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const ModelInputParams& input_params);
  void load_state_dict(const StateDict& state_dict);
};
TORCH_MODULE(FusedMoE);

}  // namespace layer
}  // namespace janus