#pragma once

#include <map>
#include <vector>

#include "core/layers/npu/npu_base_layer.h"

namespace janus {
namespace layer {

class Qwen3DecoderLoader : public BaseLoader {
 public:
  Qwen3DecoderLoader(uint64_t weight_count,
                     const ModelContext& context,
                     bool enableAddNorm);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  void merge_loaded_weights() override;

 protected:
  torch::Tensor at_placeholder_;
  bool enableAddNorm_;
  int rank_id_;
};

}  // namespace layer
}  // namespace janus