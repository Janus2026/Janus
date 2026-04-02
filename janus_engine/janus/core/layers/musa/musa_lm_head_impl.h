#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace janus {
namespace layer {
class MUSALmHeadImpl : public torch::nn::Module {
 public:
  explicit MUSALmHeadImpl(const ModelContext& context);

  ~MUSALmHeadImpl() {};

  void load_state_dict(StateDict const& state_dict);

  torch::Tensor forward(torch::Tensor const& input);

 private:
  int64_t hidden_size_;
  int64_t vocab_size_;
  torch::TensorOptions options_;
  std::vector<torch::Tensor> weights_;
};
}  // namespace layer
}  // namespace janus