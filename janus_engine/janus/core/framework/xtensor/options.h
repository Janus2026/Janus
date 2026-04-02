#pragma once

#include <torch/torch.h>

#include <vector>

#include "common/macros.h"

namespace janus {
namespace xtensor {
struct Options {
  // devices for xtensor manager pool
  PROPERTY(std::vector<torch::Device>, devices);

  // Multi-node configuration
  PROPERTY(int32_t, nnodes) = 1;
  PROPERTY(int32_t, node_rank) = 0;
};
}  // namespace xtensor
}  // namespace janus