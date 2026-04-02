#include "mlu_ops_api.h"

namespace janus::kernel::mlu {

void gather_split(const torch::Tensor& input,
                  const torch::Tensor& gather_index,
                  const torch::Tensor& valid_token_num,
                  const torch::Tensor& output_head,
                  const torch::Tensor& output_tail) {
  tmo::torch_api::gather_split(
      output_head, output_tail, input, gather_index, valid_token_num);
}

}  // namespace janus::kernel::mlu
