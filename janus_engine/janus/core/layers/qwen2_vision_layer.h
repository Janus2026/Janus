#pragma once

#include "qwen2_5_vision_layer.h"

namespace janus {
namespace layer {

class Qwen2_VisionLayerImpl : public Qwen2_5_VisionLayerImpl {
 public:
  Qwen2_VisionLayerImpl(const ModelContext& context);
  void load_state_dict(const StateDict& state_dict);
};
TORCH_MODULE(Qwen2_VisionLayer);

}  // namespace layer
}  // namespace janus