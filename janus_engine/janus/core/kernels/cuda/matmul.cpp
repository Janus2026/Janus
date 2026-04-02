#include "cuda_ops_api.h"

namespace janus::kernel::cuda {

torch::Tensor matmul(torch::Tensor a,
                     torch::Tensor b,
                     std::optional<torch::Tensor> bias) {
  namespace F = torch::nn::functional;
  return F::linear(a, b, bias.value_or(torch::Tensor()));
}

}  // namespace janus::kernel::cuda