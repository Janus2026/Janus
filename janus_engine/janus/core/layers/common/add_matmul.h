#pragma once

#include <torch/torch.h>

#include "core/framework/state_dict/state_dict.h"

namespace janus {
namespace layer {

class AddMatmulImpl : public torch::nn::Module {
 public:
  AddMatmulImpl(int64_t in,
                int64_t out,
                bool with_bias,
                const torch::TensorOptions& options);

  torch::Tensor forward(const torch::Tensor& x);

  void load_state_dict(const janus::StateDict& state_dict);

  void verify_loaded_weights(const std::string& prefix) const;

 protected:
  torch::Tensor weight_;
  torch::Tensor bias_;

  bool with_bias_;
  torch::TensorOptions options_;

  bool weight_is_loaded_ = false;
  bool bias_is_loaded_ = false;
};
TORCH_MODULE(AddMatmul);

class FusedAddMatmulImpl : public AddMatmulImpl {
 public:
  FusedAddMatmulImpl(int64_t in,
                     int64_t out,
                     bool with_bias,
                     const torch::TensorOptions& options);

  void load_state_dict(const janus::StateDict& state_dict,
                       const std::vector<std::string>& names);
};
TORCH_MODULE(FusedAddMatmul);

}  // namespace layer
}  // namespace janus
