#include "mlu_ops_api.h"

namespace janus::kernel::mlu {

void active(const torch::Tensor& input,
            torch::Tensor& output,
            const std::optional<torch::Tensor>& bias,
            const std::optional<torch::Tensor>& cusum_token_count,
            const std::string& act_mode,
            bool is_gated,
            int64_t start_expert_id,
            int64_t expert_size) {
  std::string hidden_act = act_mode;
  // TODO: act_mode gelu_pytorch_tanh not support yet.
  std::string gelu_approximate = "none";
  if (act_mode == "gelu_pytorch_tanh") {
    hidden_act = "gelu";
    gelu_approximate = "tanh";
  }
  tmo::torch_api::active(input,
                         output,
                         bias,
                         cusum_token_count,
                         hidden_act,
                         is_gated,
                         start_expert_id,
                         expert_size);
}
}  // namespace janus::kernel::mlu
