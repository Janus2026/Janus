#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace janus::kernel::npu {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
apply_moe_gating_topk_softmax(const torch::Tensor& x,
                              const std::optional<torch::Tensor>& finished,
                              int k) {
  const torch::Tensor finished_tensor =
      finished.has_value() ? finished.value() : torch::Tensor();
  return at_npu::native::custom_ops::npu_moe_gating_top_k_softmax(
      x, finished_tensor, k);
}

}  // namespace janus::kernel::npu
