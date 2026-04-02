#pragma once

#include <torch/torch.h>

#include "framework/model/model_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace janus {
namespace layer {

class MoEFusedTopkImpl : public torch::nn::Module {
 public:
  MoEFusedTopkImpl(const ModelArgs& model_args,
                   const QuantArgs& quant_args,
                   const torch::TensorOptions& options);

  std::tuple<torch::Tensor, torch::Tensor> forward(
      torch::Tensor& router_logits);

  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t topk_;
  int64_t num_expert_group_;
  int64_t topk_group_;
  double route_scale_;
  int64_t hidden_size_;
  bool renormalize_;
  std::string scoring_func_;

  DEFINE_WEIGHT(e_score_correction_bias);
};

}  // namespace layer
}  // namespace janus
