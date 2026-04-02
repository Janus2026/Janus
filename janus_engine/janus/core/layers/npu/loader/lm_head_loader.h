#pragma once

#include "base_loader.h"

namespace janus {
namespace layer {
class LmHeadLoader : public BaseLoader {
 public:
  LmHeadLoader(uint64_t weight_count, const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights(const std::string& weight_str) const override;
};
}  // namespace layer
}  // namespace janus
