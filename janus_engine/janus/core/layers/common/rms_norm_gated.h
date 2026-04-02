#pragma once

#include <torch/torch.h>

#include <optional>

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace janus {
namespace layer {

class RmsNormGatedImpl : public torch::nn::Module {
 public:
  RmsNormGatedImpl(int64_t dim,
                   double eps,
                   const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor& input,
                        std::optional<torch::Tensor> gate = std::nullopt);

  void load_state_dict(const StateDict& state_dict);

 private:
  DEFINE_WEIGHT(weight);
  int64_t norm_dim_;
  double eps_;
};
TORCH_MODULE(RmsNormGated);

}  // namespace layer
}  // namespace janus
