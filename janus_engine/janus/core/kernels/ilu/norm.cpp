#include "ilu_ops_api.h"
#include "utils.h"

using namespace ixformer;

namespace janus::kernel::ilu {

void residual_layer_norm(torch::Tensor& input,
                         torch::Tensor& output,
                         std::optional<torch::Tensor>& residual,
                         torch::Tensor& weight,
                         std::optional<torch::Tensor>& bias,
                         std::optional<torch::Tensor>& residual_out,
                         double eps) {
  auto residual_ = residual.value_or(torch::zeros_like(input));
  torch::Tensor residual_out_ = residual_out.value_or(torch::zeros_like(input));
  infer::residual_rms_norm(input,
                           residual_,
                           weight,
                           output,
                           residual_out_,
                           bias,
                           /*alpha=*/1.0,
                           eps,
                           false);
}

void rms_norm(torch::Tensor& output,
              torch::Tensor& input,
              torch::Tensor& weight,
              double eps) {
  std::optional<torch::Tensor> fused_bias = std::nullopt;
  infer::rms_norm(input, weight, output, fused_bias, eps);
}

}  // namespace janus::kernel::ilu