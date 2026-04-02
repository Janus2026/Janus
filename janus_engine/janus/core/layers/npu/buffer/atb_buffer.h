#pragma once

#include <torch/torch.h>

#include "atb/atb_infer.h"

namespace janus {

class AtbBuffer {
 public:
  explicit AtbBuffer(uint64_t bufferSize, at::Device device);
  ~AtbBuffer();
  void* get_buffer(uint64_t bufferSize);

 private:
  torch::Tensor create_attensor(uint64_t bufferSize) const;

 private:
  uint64_t buffer_size_ = 0;
  torch::Tensor at_tensor_;
  at::Device device_;

  at::TensorOptions options_;
};

}  // namespace janus
