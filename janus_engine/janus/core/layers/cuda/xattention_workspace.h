#pragma once

#include <torch/torch.h>

#include <cstdint>

#include "core/common/macros.h"

namespace janus::layer::xattention {

class XAttentionWorkspace {
 public:
  static XAttentionWorkspace& get_instance() {
    thread_local XAttentionWorkspace instance;
    return instance;
  };

  void initialize(const torch::Device& device);

  torch::Tensor get_int_workspace_buffer();

  torch::Tensor get_page_locked_int_workspace_buffer();

 private:
  XAttentionWorkspace() = default;
  ~XAttentionWorkspace() = default;
  DISALLOW_COPY_AND_ASSIGN(XAttentionWorkspace);

  torch::Tensor int_workspace_buffer_;
  torch::Tensor page_locked_int_workspace_buffer_;
};

}  // namespace janus::layer::xattention
