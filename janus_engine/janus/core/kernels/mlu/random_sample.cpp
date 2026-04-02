#include "mlu_ops_api.h"

namespace janus::kernel::mlu {

torch::Tensor random_sample(const torch::Tensor& probs) {
  torch::Tensor flat_probs;
  if (probs.dim() == 3) {
    flat_probs = probs.reshape({-1, probs.size(2)});
  } else {
    flat_probs = probs;
  }
  auto output =
      torch::empty({flat_probs.size(0), 1},
                   torch::dtype(torch::kInt64).device(probs.device()));
  tmo::torch_api::random_sample(flat_probs, output, true, torch::Generator());
  if (probs.dim() == 3) {
    return output.reshape({probs.size(0), probs.size(1)});
  }
  return output.flatten();
}

}  // namespace janus::kernel::mlu