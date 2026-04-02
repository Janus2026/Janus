#pragma once

#include <torch/torch.h>

#include "framework/state_dict/state_dict.h"
#include "musa_layer_base.h"

namespace janus {
namespace layer {
class MusaMLPImpl : public MUSALayerBaseImpl {
 public:
  MusaMLPImpl(int32_t hidden_size,
              int32_t intermediate_size,
              bool is_gated,
              bool has_bias,
              const std::string& hidden_act,
              const QuantArgs& quant_args,
              const ParallelArgs& parallel_args,
              const torch::TensorOptions& options,
              float rms_eps);
  ~MusaMLPImpl() {};

  torch::Tensor forward(torch::Tensor& input,
                        ForwardParams& fwd_params) override;

  void load_state_dict(StateDict const& state_dict) override;

 private:
  // todo: add member
  int32_t hidden_size_;
  int32_t intermediate_size_;
  float rms_eps;
  constexpr static int32_t weight_num_ = 4;
};
TORCH_MODULE(MusaMLP);

}  // namespace layer
}  // namespace janus