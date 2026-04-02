#pragma once

#include <map>
#include <vector>

#include "base_loader.h"

namespace janus {
namespace layer {

class Glm4VisionEncoderLoader : public BaseLoader {
 public:
  Glm4VisionEncoderLoader(uint64_t weight_count, const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  void merge_loaded_weights() override;

 private:
  void get_weights_col_packed_qkv();
};

}  // namespace layer
}  // namespace janus
