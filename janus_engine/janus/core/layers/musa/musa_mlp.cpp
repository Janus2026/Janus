#include "musa_mlp.h"

#include <cstdint>

#include "MTTOplib/FusedMLP.h"
#include "MTTOplib/WeightReorder.h"

namespace janus {
namespace layer {
MusaMLPImpl::MusaMLPImpl(int32_t hidden_size,
                         int32_t intermediate_size,
                         bool is_gated,
                         bool has_bias,
                         const std::string& hidden_act,
                         const QuantArgs& quant_args,
                         const ParallelArgs& parallel_args,
                         const torch::TensorOptions& options,
                         float rms_eps)
    : MUSALayerBaseImpl(options),
      hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      rms_eps(rms_eps) {
  weights_.resize(weight_num_);
}

torch::Tensor MusaMLPImpl::forward(torch::Tensor& input,
                                   ForwardParams& fwd_params) {
  return janus_musa::FusedMLP(input, weights_, rms_eps);
}

void MusaMLPImpl::load_state_dict(StateDict const& state_dict) {
  using WeightMeta = std::pair<std::string, std::vector<int64_t>>;
  static int32_t all_loaded = 0;
  std::vector<WeightMeta> meta = {
      {"up_proj.", {intermediate_size_, hidden_size_}},
      {"gate_proj.", {intermediate_size_, hidden_size_}},
      {"down_proj.", {hidden_size_, intermediate_size_}}};
  for (int32_t i = 0; i < meta.size(); ++i) {
    all_loaded += load_weight_common(
        state_dict.get_dict_with_prefix("mlp." + meta[i].first),
        meta[i].second,
        i);
  }
  all_loaded += load_weight_common(
      state_dict.get_dict_with_prefix("post_attention_layernorm."),
      {hidden_size_},
      3);
  if (all_loaded == weight_num_) {
    all_loaded = 0;
    weights_ = janus_musa::ReorderMLP(weights_);
  }
}
}  // namespace layer
}  // namespace janus
