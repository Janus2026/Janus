#include "cuda_ops_api.h"

namespace janus::kernel::cuda {

torch::Tensor fp8_scaled_matmul(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_scale,
    torch::ScalarType output_dtype,
    const std::optional<torch::Tensor>& bias /* = std::nullopt */,
    const std::optional<torch::Tensor>& output /* = std::nullopt */) {
  // Prepare output tensor
  torch::Tensor result_output;
  if (output.has_value() && output.value().defined()) {
    result_output = output.value();
  } else {
    result_output =
        torch::empty({a.size(0), b.size(0)}, a.options().dtype(output_dtype));
  }

  // Transpose weight for CUTLASS: [N, K] -> [K, N] (column-major)
  // NOTE: Do NOT call .contiguous() - .t() makes it column-major (stride(0)==1)
  auto b_t = b.t();

  // Call underlying kernel
  cutlass_scaled_mm(result_output, a, b_t, a_scale, b_scale, bias);

  return result_output;
}

}  // namespace janus::kernel::cuda
