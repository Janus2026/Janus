#pragma once
#include <torch/torch.h>

#include <string>
#include <unordered_map>

namespace janus {

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

struct CacheStepIn {
  int64_t step_id = 0;
  TensorMap tensors;

  CacheStepIn(int64_t step_id) : step_id(step_id) {}
  CacheStepIn(int64_t step_id, const TensorMap& tensors)
      : step_id(step_id), tensors(tensors) {}
};

struct CacheStepOut {
  TensorMap tensors;
  CacheStepOut(const TensorMap& tensors) : tensors(tensors) {}
};

struct CacheBlockIn {
  int64_t block_id = 0;
  TensorMap tensors;

  CacheBlockIn(int64_t block_id) : block_id(block_id) {}
  CacheBlockIn(int64_t block_id, const TensorMap& tensors)
      : block_id(block_id), tensors(tensors) {}
};

struct CacheBlockOut {
  TensorMap tensors;
  CacheBlockOut(const TensorMap& tensors) : tensors(tensors) {}
};

}  // namespace janus
