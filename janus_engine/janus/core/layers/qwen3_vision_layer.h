#pragma once
#include "qwen2_5_vision_layer.h"

namespace janus {
namespace layer {

class Qwen3_VisionLayerImpl : public Qwen2_5_VisionLayerImpl {
 public:
  Qwen3_VisionLayerImpl(const ModelContext& context);

  void load_state_dict(const StateDict& state_dict);
};
TORCH_MODULE(Qwen3_VisionLayer);

}  // namespace layer
}  // namespace janus