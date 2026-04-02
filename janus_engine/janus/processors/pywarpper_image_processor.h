#pragma once

#include <vector>

#include "image_processor.h"

namespace janus {

struct MMData;

class PyWarpperImageProcessor : public ImageProcessor {
 public:
  PyWarpperImageProcessor(const ModelArgs&);
  ~PyWarpperImageProcessor() override = default;

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;
};

}  // namespace janus
