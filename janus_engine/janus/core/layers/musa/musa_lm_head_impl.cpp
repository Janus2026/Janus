#include "musa_lm_head_impl.h"

#include "MTTOplib/Ops.h"
#include "MTTOplib/WeightReorder.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace janus {
namespace layer {
MUSALmHeadImpl::MUSALmHeadImpl(const ModelContext& context)
    : options_(context.get_tensor_options()),
      hidden_size_(context.get_model_args().hidden_size()),
      vocab_size_(context.get_model_args().vocab_size()) {}

void MUSALmHeadImpl::load_state_dict(StateDict const& state_dict) {
  if (state_dict.size() == 0) return;
  DEFINE_WEIGHT(weight);
  weight_ = torch::empty({vocab_size_, hidden_size_}, options_);
  LOAD_WEIGHT(weight);
  if (weight_is_loaded_) {
    weights_.emplace_back(weight_.clone());
    weights_ = janus_musa::ReorderLMHead(weights_);
  }
}

torch::Tensor MUSALmHeadImpl::forward(torch::Tensor const& input) {
  torch::Tensor out =
      torch::empty({input.size(0), weights_[0].size(0)}, input.options());
  janus_musa::Matmul(input, out, weights_[0]);
  return out;
}
}  // namespace layer
}  // namespace janus
