#pragma once

#include <torch/torch.h>

#include <string>

#include "core/framework/request/mm_data.h"

namespace janus {

class InputProcessor {
 public:
  virtual ~InputProcessor() = default;

  virtual void process(std::string& prompt, const MMData& mm_data) = 0;
  virtual void find_mm_spans(const std::vector<int>& prompt, MMData& mm_data) {
  };
};

}  // namespace janus
