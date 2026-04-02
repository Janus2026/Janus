#pragma once

#include <map>
#include <vector>

#include "core/layers/npu/npu_base_layer.h"

namespace janus {
namespace layer {

class WordEmbeddingLoader : public BaseLoader {
 public:
  WordEmbeddingLoader(uint64_t weight_count, const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights(const std::string& prefix) const override;
};

}  // namespace layer
}  // namespace janus