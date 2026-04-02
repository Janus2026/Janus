#pragma once

#include <map>
#include <vector>

#include "base_manual_loader.h"
#include "core/layers/npu/npu_base_layer.h"

namespace janus {
namespace layer {

class WordEmbeddingManualLoader : public BaseManualLoader {
 public:
  WordEmbeddingManualLoader(uint64_t weight_count, const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights(const std::string& prefix) const override;

 protected:
  void merge_host_at_weights() override;
};

}  // namespace layer
}  // namespace janus