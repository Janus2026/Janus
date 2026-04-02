#include "kernels/cuda/cuda_ops_api.h"
#include "moe_topk_sigmoid_kernels.cuh"
#include "moe_topk_softmax_kernels.cuh"

namespace janus::kernel::cuda {

std::tuple<torch::Tensor, torch::Tensor> moe_fused_topk(
    torch::Tensor& gating_output,
    int64_t topk,
    bool renormalize,
    const std::optional<torch::Tensor>& correction_bias,
    const std::string& scoring_func) {
  int64_t num_tokens = gating_output.size(0);

  torch::Tensor topk_weights = torch::empty(
      {num_tokens, topk},
      torch::dtype(torch::kFloat32).device(gating_output.device()));
  torch::Tensor topk_ids =
      torch::empty({num_tokens, topk},
                   torch::dtype(torch::kInt32).device(gating_output.device()));

  if (scoring_func == "softmax") {
    std::optional<torch::Tensor> none_correction_bias = std::nullopt;
    topk_softmax(topk_weights,
                 topk_ids,
                 gating_output,
                 renormalize,
                 /*moe_softcapping=*/0.0,
                 none_correction_bias);
  } else if (scoring_func == "sigmoid") {
    topk_sigmoid(
        topk_weights, topk_ids, gating_output, renormalize, correction_bias);
  } else {
    LOG(FATAL) << "Unsupported scoring function for moe topk: " << scoring_func
               << "only softmax and sigmoid are supported";
  }

  return std::make_tuple(topk_weights, topk_ids);
}

}  // namespace janus::kernel::cuda
