#pragma once

#include "base_manual_loader.h"

namespace janus {
namespace layer {
class LmHeadManualLoader : public BaseManualLoader {
 public:
  LmHeadManualLoader(uint64_t weight_count, const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights(const std::string& weight_str) const override;

 protected:
  void merge_host_at_weights() override;
};
}  // namespace layer
}  // namespace janus