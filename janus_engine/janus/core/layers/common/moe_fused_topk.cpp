#include "moe_fused_topk.h"

#include "kernels/ops_api.h"

namespace janus {
namespace layer {

MoEFusedTopkImpl::MoEFusedTopkImpl(const ModelArgs& model_args,
                                   const QuantArgs& quant_args,
                                   const torch::TensorOptions& options)
    : topk_(model_args.num_experts_per_tok()),
      num_expert_group_(model_args.n_group()),
      topk_group_(model_args.topk_group()),
      route_scale_(model_args.routed_scaling_factor()),
      hidden_size_(model_args.hidden_size()),
      renormalize_(model_args.norm_topk_prob()),
      scoring_func_(model_args.scoring_func()) {
  const std::string& topk_method = model_args.topk_method();
  if (topk_method == "noaux_tc") {
    e_score_correction_bias_ = register_parameter(
        "e_score_correction_bias",
        torch::empty({model_args.n_routed_experts()}, options),
        false);
  }
}

// select the experts and return the reduce_weight and expert_id
std::tuple<torch::Tensor, torch::Tensor> MoEFusedTopkImpl::forward(
    torch::Tensor& router_logits) {
  std::optional<torch::Tensor> e_score_correction_bias = std::nullopt;
  if (e_score_correction_bias_.defined()) {
    e_score_correction_bias = e_score_correction_bias_;
  }

  janus::kernel::MoeFusedTopkParams moe_active_topk_params;
  moe_active_topk_params.input = router_logits;
  moe_active_topk_params.topk = topk_;
  moe_active_topk_params.num_expert_group = num_expert_group_;
  moe_active_topk_params.topk_group = topk_group_;
  moe_active_topk_params.normalize = renormalize_;
  moe_active_topk_params.normed_by = "topk_logit";
  moe_active_topk_params.scoring_func = scoring_func_;
  moe_active_topk_params.route_scale = route_scale_;
  moe_active_topk_params.e_score_correction_bias = e_score_correction_bias;

  return janus::kernel::moe_active_topk(moe_active_topk_params);
}

void MoEFusedTopkImpl::load_state_dict(const StateDict& state_dict) {
  if (e_score_correction_bias_.defined() &&
      !e_score_correction_bias_is_loaded_) {
    LOAD_WEIGHT(e_score_correction_bias);
  }
}
}  // namespace layer
}  // namespace janus
