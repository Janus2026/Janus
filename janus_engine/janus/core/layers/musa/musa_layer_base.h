#pragma once

#include <torch/torch.h>

#include "MTTOplib/Attention.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/attention_metadata.h"

namespace janus::layer {

struct ForwardParams {
  torch::Tensor& positions;
  AttentionMetadata const& attn_meta;
  KVCache& kv_cache;
  ModelInputParams const& input_params;
};

class MUSALayerBaseImpl : public torch::nn::Module {
 public:
  MUSALayerBaseImpl() = default;
  explicit MUSALayerBaseImpl(torch::TensorOptions const& options)
      : options_(options) {};
  virtual ~MUSALayerBaseImpl() = default;

  bool load_weight_common(StateDict const& state_dict,
                          std::vector<int64_t> const& shape,
                          int32_t idx) {
    if (state_dict.size() == 0) {
      return false;
    }
    DEFINE_WEIGHT(weight);
    weight_ = torch::empty(shape, options_);
    LOAD_WEIGHT(weight);
    if (weight_is_loaded_) {
      weights_[idx] = weight_.clone();
    }
    return weight_is_loaded_;
  }

  virtual void load_state_dict(const StateDict& state_dict) = 0;
  virtual torch::Tensor forward(torch::Tensor& input,
                                ForwardParams& fwd_params) = 0;

 protected:
  std::vector<torch::Tensor> weights_;
  torch::TensorOptions options_;
};
TORCH_MODULE(MUSALayerBase);

}  // namespace janus::layer