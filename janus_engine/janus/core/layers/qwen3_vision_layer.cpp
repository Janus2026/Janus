#include "qwen3_vision_layer.h"

namespace janus {
namespace layer {

Qwen3_VisionLayerImpl::Qwen3_VisionLayerImpl(const ModelContext& context)
    : Qwen2_5_VisionLayerImpl(context, true) {}

void Qwen3_VisionLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
  mlp_->load_state_dict(
      state_dict.get_dict_with_prefix("mlp."), {"linear_fc1."}, "linear_fc2.");
  norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
  norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
}

}  // namespace layer
}  // namespace janus