#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace janus::kernel::npu {

torch::Tensor rms_norm(const torch::Tensor& input,
                       const torch::Tensor& weight,
                       double eps,
                       const std::string& mode) {
  if (mode != "rmsnorm") {
    LOG(FATAL) << "Only rmsnorm mode is supported in NPU rms_norm";
  }
  std::tuple<at::Tensor, at::Tensor> result =
      at_npu::native::custom_ops::npu_rms_norm(input, weight, eps);
  auto normalized_input = std::get<0>(result);
  return normalized_input;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> add_rms_norm(
    const torch::Tensor& x1,
    const torch::Tensor& x2,
    const torch::Tensor& gamma,
    double epsilon) {
  return at_npu::native::custom_ops::npu_add_rms_norm(x1, x2, gamma, epsilon);
}

}  // namespace janus::kernel::npu