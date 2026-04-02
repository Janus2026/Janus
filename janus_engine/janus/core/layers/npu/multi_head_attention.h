#pragma once

#include <torch/torch.h>

#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace janus {
namespace layer {

class MultiheadAttentionImpl : public torch::nn::Module {
 public:
  MultiheadAttentionImpl(const ModelContext& context);

  torch::Tensor forward(torch::Tensor query,
                        torch::Tensor key,
                        torch::Tensor value,
                        torch::Tensor key_padding_mask);

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights(const std::string& prefix) const;

 private:
  int64_t n_head_;
  int64_t head_dim_;
  int64_t hidden_size_;
  torch::TensorOptions options_;

  DEFINE_WEIGHT(in_proj_weight);
  DEFINE_WEIGHT(in_proj_bias);
  DEFINE_WEIGHT(out_proj_weight);
  DEFINE_WEIGHT(out_proj_bias);
};

TORCH_MODULE(MultiheadAttention);

}  // namespace layer
}  // namespace janus
