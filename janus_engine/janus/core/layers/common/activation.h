#pragma once

#include <torch/torch.h>

#include <string>

namespace janus {
namespace layer {

class ActivationImpl : public torch::nn::Module {
 public:
  ActivationImpl(const std::string& act_mode, bool is_gated);

  void forward(torch::Tensor& input, torch::Tensor& output);

 private:
  std::string act_mode_;
  bool is_gated_;
};
TORCH_MODULE(Activation);

}  // namespace layer
}  // namespace janus