#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace janus::kernel::npu {

torch::Tensor active(const torch::Tensor& input, const std::string& act_mode) {
  if (act_mode != "silu" && act_mode != "swiglu") {
    LOG(FATAL) << "Only swiglu activation is supported in NPU active";
  }
  return at_npu::native::custom_ops::npu_swiglu(input);
}
}  // namespace janus::kernel::npu