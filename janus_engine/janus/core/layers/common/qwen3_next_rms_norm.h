#pragma once

#include <torch/torch.h>

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace janus {
namespace layer {

class Qwen3NextRMSNormImpl : public torch::nn::Module {
 public:
  Qwen3NextRMSNormImpl(int64_t dim,
                       double eps,
                       const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor& input);

  void load_state_dict(const StateDict& state_dict);

 private:
  DEFINE_WEIGHT(weight);
  int64_t norm_dim_;
  double eps_;
};
TORCH_MODULE(Qwen3NextRMSNorm);

}  // namespace layer
}  // namespace janus
