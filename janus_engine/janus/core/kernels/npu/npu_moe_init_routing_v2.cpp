#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace janus::kernel::npu {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
apply_npu_moe_init_routing_v2(const torch::Tensor& x,
                              const torch::Tensor& expert_idx,
                              const std::optional<torch::Tensor>& scale,
                              const std::optional<torch::Tensor>& offset,
                              int active_num,
                              int expert_capacity,
                              int expert_num,
                              int drop_pad_mode,
                              int expert_tokens_num_type,
                              bool expert_tokens_num_flag,
                              int quant_mode,
                              torch::IntArrayRef active_expert_range,
                              int row_idx_type) {
  return at_npu::native::custom_ops::npu_moe_init_routing_v2(
      x,
      expert_idx,
      c10::nullopt,
      c10::nullopt,
      active_num,
      expert_capacity,
      expert_num,
      0,
      expert_tokens_num_type,
      expert_tokens_num_flag,
      quant_mode,
      active_expert_range,
      0);
}

}  // namespace janus::kernel::npu
