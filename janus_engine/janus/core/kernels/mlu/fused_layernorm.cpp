#include "mlu_ops_api.h"

namespace janus::kernel::mlu {

void fused_layernorm(const torch::Tensor& input,
                     torch::Tensor& output,
                     const std::optional<torch::Tensor>& residual,
                     const torch::Tensor& weight,
                     const std::optional<torch::Tensor>& beta,
                     const std::optional<torch::Tensor>& bias,
                     const std::optional<torch::Tensor>& quant_scale,
                     const std::optional<torch::Tensor>& residual_out,
                     const std::optional<torch::Tensor>& smooth_quant_scale,
                     const std::optional<torch::Tensor>& normed_out,
                     const std::string& mode,
                     double eps,
                     bool store_output_before_norm,
                     bool store_output_after_norm,
                     bool dynamic_quant) {
  tmo::torch_api::fused_layernorm(input,
                                  output,
                                  residual,
                                  weight,
                                  beta,
                                  bias,
                                  quant_scale,
                                  residual_out,
                                  smooth_quant_scale,
                                  normed_out,
                                  mode,
                                  eps,
                                  store_output_before_norm,
                                  store_output_after_norm,
                                  dynamic_quant);
}

}  // namespace janus::kernel::mlu