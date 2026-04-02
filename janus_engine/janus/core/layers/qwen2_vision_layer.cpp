#include "qwen2_vision_layer.h"

namespace janus {
namespace layer {

Qwen2_VisionLayerImpl::Qwen2_VisionLayerImpl(const ModelContext& context)
    : Qwen2_5_VisionLayerImpl(context, true) {}

void Qwen2_VisionLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
  mlp_->load_state_dict(
      state_dict.get_dict_with_prefix("mlp."), {"fc1."}, "fc2.");
  norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
  norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
}

}  // namespace layer
}  // namespace janus