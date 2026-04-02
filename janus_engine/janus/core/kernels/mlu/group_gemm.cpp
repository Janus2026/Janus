#include "mlu_ops_api.h"

namespace janus::kernel::mlu {

torch::Tensor group_gemm(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& token_count,
                         torch::Tensor& output,
                         const std::optional<torch::Tensor>& a_scale,
                         const std::optional<torch::Tensor>& b_scale,
                         const std::optional<torch::List<int64_t>>& quant_flag,
                         const int64_t max_dim,
                         const bool trans_a,
                         const bool trans_b,
                         const int64_t a_quant_bit) {
  tmo::torch_api::group_gemm(a,
                             b,
                             token_count,
                             output,
                             /*gather_idx=*/std::nullopt,
                             /*c=*/std::nullopt,
                             /*alpha=*/std::nullopt,
                             /*beta=*/std::nullopt,
                             a_scale,
                             b_scale,
                             /*bias=*/std::nullopt,
                             /*a_calibration=*/std::nullopt,
                             /*b_calibration=*/std::nullopt,
                             quant_flag,
                             /*b_offset=*/std::nullopt,
                             /*tile_config=*/std::nullopt,
                             max_dim,
                             trans_a,
                             trans_b,
                             a_quant_bit);
  return output;
}

}  // namespace janus::kernel::mlu
