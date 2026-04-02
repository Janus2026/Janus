#include "xattention_workspace.h"

#include <glog/logging.h>

namespace janus::layer::xattention {

void XAttentionWorkspace::initialize(const torch::Device& device) {
  const bool is_already_initialized =
      int_workspace_buffer_.defined() &&
      page_locked_int_workspace_buffer_.defined() &&
      int_workspace_buffer_.device() == device;
  if (is_already_initialized) {
    return;
  }

  LOG(INFO) << "XAttentionWorkspace initialize on device: " << device;

  int_workspace_buffer_ = torch::empty(
      {8 * 1024 * 1024}, torch::dtype(torch::kUInt8).device(device));
  page_locked_int_workspace_buffer_ = torch::empty(
      {int_workspace_buffer_.size(0)},
      torch::dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
}

torch::Tensor XAttentionWorkspace::get_int_workspace_buffer() {
  return int_workspace_buffer_;
}

torch::Tensor XAttentionWorkspace::get_page_locked_int_workspace_buffer() {
  return page_locked_int_workspace_buffer_;
}

}  // namespace janus::layer::xattention
