#pragma once

#include <map>
#include <vector>

#include "base_loader.h"

namespace janus {
namespace layer {

class RMSNormLoader : public BaseLoader {
 public:
  RMSNormLoader(uint64_t weight_count, const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;

  void verify_loaded_weights(const std::string& weight_str) const override;

 protected:
  int rank_id_;
  torch::ScalarType dtype_;
};

}  // namespace layer
}  // namespace janus