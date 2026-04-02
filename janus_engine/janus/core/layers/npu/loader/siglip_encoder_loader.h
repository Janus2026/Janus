#pragma once

#include <map>
#include <vector>

#include "base_loader.h"

namespace janus {
namespace layer {

class SiglipEncoderUpLoader : public BaseLoader {
 public:
  explicit SiglipEncoderUpLoader(const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
};

class SiglipEncoderDownLoader : public BaseLoader {
 public:
  explicit SiglipEncoderDownLoader(const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
};

}  // namespace layer
}  // namespace janus