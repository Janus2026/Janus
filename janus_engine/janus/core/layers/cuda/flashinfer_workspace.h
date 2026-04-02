#pragma once

#include <torch/torch.h>

#include <cstdint>

#include "core/common/macros.h"

namespace janus::layer::flashinfer {

class FlashinferWorkspace {
 public:
  static FlashinferWorkspace& get_instance() {
    thread_local FlashinferWorkspace instance;
    return instance;
  };

  void initialize(const torch::Device& device);

  torch::Tensor get_float_workspace_buffer();
  torch::Tensor get_int_workspace_buffer();
  torch::Tensor get_page_locked_int_workspace_buffer();

 private:
  FlashinferWorkspace() = default;
  ~FlashinferWorkspace() = default;
  DISALLOW_COPY_AND_ASSIGN(FlashinferWorkspace);

  torch::Tensor float_workspace_buffer_;
  torch::Tensor int_workspace_buffer_;
  torch::Tensor page_locked_int_workspace_buffer_;
};

}  // namespace janus::layer::flashinfer