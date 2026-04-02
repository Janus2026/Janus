#include "cuda_ops_api.h"

namespace janus::kernel::cuda {

std::tuple<torch::Tensor, torch::Tensor> fp8_scaled_quantize(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& output /* = std::nullopt */,
    const std::optional<torch::Tensor>& scale /* = std::nullopt */) {
  // Prepare output tensor
  torch::Tensor result_output;
  if (output.has_value() && output.value().defined()) {
    result_output = output.value();
  } else {
    result_output =
        torch::empty_like(input, input.options().dtype(torch::kFloat8_e4m3fn));
  }

  torch::Tensor result_scale;
  if (scale.has_value() && scale.value().defined()) {
    // Static quantization - use pre-computed scale
    result_scale = scale.value();
  } else {
    // Dynamic quantization - compute scale on the fly
    // 448 is max value for FP8 e4m3
    auto amax = input.abs().max();
    result_scale = (amax / 448.0f).clamp_min(1e-12f).to(torch::kFloat32);
  }

  // Call underlying kernel
  static_scaled_fp8_quant(result_output, input, result_scale);

  return std::make_tuple(result_output, result_scale);
}

}  // namespace janus::kernel::cuda
