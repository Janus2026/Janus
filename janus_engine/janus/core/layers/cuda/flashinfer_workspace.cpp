#include "flashinfer_workspace.h"

#include <glog/logging.h>

#include "core/common/global_flags.h"

namespace janus::layer::flashinfer {

void FlashinferWorkspace::initialize(const torch::Device& device) {
  LOG(INFO) << "FlashinferWorkspace initialize on device: " << device;
  float_workspace_buffer_ =
      torch::empty({FLAGS_flashinfer_workspace_buffer_size},
                   torch::dtype(torch::kUInt8).device(device));
  int_workspace_buffer_ = torch::empty(
      {8 * 1024 * 1024}, torch::dtype(torch::kUInt8).device(device));
  page_locked_int_workspace_buffer_ = torch::empty(
      {int_workspace_buffer_.size(0)},
      torch::dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
  LOG(INFO) << "FlashinferWorkspace initialize end with "
               "flashinfer_workspace_buffer_size: "
            << FLAGS_flashinfer_workspace_buffer_size;
}

torch::Tensor FlashinferWorkspace::get_float_workspace_buffer() {
  return float_workspace_buffer_;
}

torch::Tensor FlashinferWorkspace::get_int_workspace_buffer() {
  return int_workspace_buffer_;
}

torch::Tensor FlashinferWorkspace::get_page_locked_int_workspace_buffer() {
  return page_locked_int_workspace_buffer_;
}

}  // namespace janus::layer::flashinfer