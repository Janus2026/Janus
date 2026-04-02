#pragma once

#include <torch/torch.h>

#include "core/framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace janus {
namespace layer {

class RMSNormImpl : public torch::nn::Module {
 public:
  RMSNormImpl(int64_t dim, double eps, const torch::TensorOptions& options);
  RMSNormImpl(const ModelContext& context);

  // Standard forward: returns (normalized_output, updated_residual)
  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      torch::Tensor& input,
      std::optional<torch::Tensor> residual = std::nullopt,
      std::optional<torch::Tensor> inplace_output = std::nullopt);

  // Fused forward with FP8 quantization output (for static quantization)
  // Returns: (fp8_quantized_output, updated_residual)
  // This combines RMSNorm + FP8 quantization to reduce memory bandwidth
  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward_fp8(
      torch::Tensor& input,
      const torch::Tensor& fp8_scale,
      std::optional<torch::Tensor> residual = std::nullopt);

  void set_layernorm_mode();

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor weight() const { return weight_; }
  torch::Tensor bias() const { return bias_; }
  double eps() const { return eps_; }

 private:
  DEFINE_WEIGHT(weight);
  DEFINE_WEIGHT(bias);
  int64_t norm_dim_;
  double eps_;
  std::string mode_;
};
TORCH_MODULE(RMSNorm);

}  // namespace layer
}  // namespace janus
