#include "mlu_ops_api.h"

namespace janus::kernel::mlu {

torch::Tensor matmul(const torch::Tensor& a,
                     const torch::Tensor& b,
                     const std::optional<torch::Tensor>& bias,
                     const std::optional<torch::Tensor>& c,
                     double alpha,
                     double beta) {
  return tmo::torch_api::matmul(a,
                                b,
                                bias,
                                c,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                "none",
                                alpha,
                                beta,
                                true,
                                true,
                                1.0,
                                1.0,
                                false,
                                true);
}

}  // namespace janus::kernel::mlu
