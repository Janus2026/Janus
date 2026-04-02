#pragma once

#include <map>
#include <vector>

#include "base_loader.h"

namespace janus {
namespace layer {

class Glm4DecoderLoader : public BaseLoader {
 public:
  Glm4DecoderLoader(uint64_t weight_count, const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  void merge_loaded_weights() override;

 private:
  at::Tensor at_placeholder_;
};

}  // namespace layer
}  // namespace janus
