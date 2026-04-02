#include "ilu_ops_api.h"

namespace janus::kernel::ilu {

torch::Tensor group_gemm(torch::Tensor& input,
                         torch::Tensor& weight,
                         torch::Tensor& tokens_per_experts,
                         const std::optional<torch::Tensor>& dst_to_src,
                         torch::Tensor& output) {
  infer::moe_w16a16_group_gemm(
      output,
      input,
      weight,
      tokens_per_experts,
      dst_to_src,
      /*bias=*/std::nullopt,
      /*format=*/"TN",
      /*persistent=*/0,
      /*output_n=*/tokens_per_experts.sum().item<int64_t>());

  return output;
}

}  // namespace janus::kernel::ilu
