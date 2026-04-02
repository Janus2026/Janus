#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace janus::kernel::npu {

torch::Tensor apply_npu_moe_token_unpermute(
    const torch::Tensor& permuted_tokens,
    const torch::Tensor& sorted_indices,
    const std::optional<torch::Tensor>& probes,
    bool padded_mode,
    c10::OptionalIntArrayRef restore_shape) {
  if (!padded_mode) {
    return at_npu::native::custom_ops::npu_moe_token_unpermute(
        permuted_tokens, sorted_indices, probes.value());
  } else {
    return at_npu::native::custom_ops::npu_moe_token_unpermute(
        permuted_tokens,
        sorted_indices,
        probes.value(),
        padded_mode,
        restore_shape.value());
  }
}

}  // namespace janus::kernel::npu
