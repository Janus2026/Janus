#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace janus::kernel::npu {

torch::Tensor matmul(const torch::Tensor& a,
                     const torch::Tensor& b,
                     const std::optional<torch::Tensor>& bias) {
  if (!bias.has_value()) {
    return torch::nn::functional::linear(a, b);
  } else {
    return torch::nn::functional::linear(a, b, bias.value());
  }
}

}  // namespace janus::kernel::npu
