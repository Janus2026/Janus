#include "moe_gate.h"

#include "kernels/ops_api.h"

namespace janus {
namespace layer {

MoEGateImpl::MoEGateImpl(const ModelArgs& model_args,
                         const QuantArgs& quant_args,
                         const torch::TensorOptions& options)
    : num_experts_(model_args.n_routed_experts()),
      topk_(model_args.num_experts_per_tok()),
      num_expert_group_(model_args.n_group()),
      topk_group_(model_args.topk_group()),
      route_scale_(static_cast<double>(model_args.routed_scaling_factor())),
      hidden_size_(model_args.hidden_size()),
      renormalize_(model_args.norm_topk_prob() ? 1 : 0),
      scoring_func_(model_args.scoring_func()) {
  const std::string& topk_method = model_args.topk_method();
  if (topk_method == "noaux_tc") {
    e_score_correction_bias_ = register_parameter(
        "e_score_correction_bias",
        torch::empty({model_args.n_routed_experts()}, options),
        false);
  }

  gate_ = register_module("gate_proj",
                          ReplicatedLinear(model_args.hidden_size(),
                                           model_args.n_routed_experts(),
                                           false,
                                           quant_args,
                                           options));
}

std::tuple<torch::Tensor, torch::Tensor> MoEGateImpl::forward(
    torch::Tensor& hidden_states) {
  torch::Tensor router_logits = gate_->forward(hidden_states);
  torch::Tensor router_logits_2d =
      router_logits.reshape({-1, router_logits.size(-1)});

  std::optional<torch::Tensor> e_score_correction_bias = std::nullopt;
  if (e_score_correction_bias_.defined()) {
    e_score_correction_bias = e_score_correction_bias_;
  }

  janus::kernel::MoeFusedTopkParams moe_active_topk_params;
  moe_active_topk_params.input = router_logits_2d;
  moe_active_topk_params.topk = topk_;
  moe_active_topk_params.num_expert_group = num_expert_group_;
  moe_active_topk_params.topk_group = topk_group_;
  moe_active_topk_params.normalize = static_cast<bool>(renormalize_);
  moe_active_topk_params.normed_by = "topk_logit";
  moe_active_topk_params.scoring_func = scoring_func_;
  moe_active_topk_params.route_scale = route_scale_;
  moe_active_topk_params.e_score_correction_bias = e_score_correction_bias;

  return janus::kernel::moe_active_topk(moe_active_topk_params);
}

void MoEGateImpl::load_state_dict(const StateDict& state_dict) {
  gate_->load_state_dict(state_dict);
  if (e_score_correction_bias_.defined() &&
      !e_score_correction_bias_is_loaded_) {
    LOAD_WEIGHT(e_score_correction_bias);
  }
}

}  // namespace layer
}  // namespace janus
