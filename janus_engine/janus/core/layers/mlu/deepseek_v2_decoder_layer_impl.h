#pragma once

#include <torch/torch.h>

#include <optional>

#include "attention.h"
#include "deepseek_v2_attention.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/dense_mlp.h"
#include "layers/common/dp_utils.h"
#include "layers/common/rms_norm.h"
#include "layers/mlu/deepseek_v32_sp_context.h"
#include "layers/mlu/fused_moe.h"

namespace janus {
namespace layer {

class DeepseekV2DecoderLayerTestPeer;

class DeepseekV2DecoderLayerImpl : public torch::nn::Module {
 public:
  explicit DeepseekV2DecoderLayerImpl(const ModelContext& context,
                                      int32_t layer_id);

  ~DeepseekV2DecoderLayerImpl() override = default;

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;

  void set_sequence_parallel_context(
      const v32_sp::DeepseekV32SPContext* sp_ctx) {
    sequence_parallel_context_ = sp_ctx;
  }

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  enum class PostAttnMode {
    kReplicated,
    kPackedLocal,
    kDpGather,
    kTpPadded,
  };

  struct PostAttnCarrier {
    torch::Tensor ffn_in;
    torch::Tensor skip_local;
    PaddingInfo pad_info;
    PostAttnMode mode = PostAttnMode::kReplicated;
  };

  PostAttnCarrier build_post_attn_carrier(
      torch::Tensor x,
      const torch::Tensor& residual,
      const ModelInputParams& input_params,
      DeepseekV2AttentionImpl::PostAttnLayout attn_layout,
      bool need_dp_gather,
      bool enable_moe_all2all);

  torch::Tensor materialize_ffn_input(const PostAttnCarrier& carrier,
                                      const ModelInputParams& input_params);

  torch::Tensor restore_ffn_output(torch::Tensor x,
                                   const PostAttnCarrier& carrier,
                                   const ModelInputParams& input_params);

  friend class DeepseekV2DecoderLayerTestPeer;

  // parallel args
  ParallelArgs parallel_args_;
  bool enable_deep_ep_;
  bool is_moe_layer_;

  DeepseekV2Attention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  FusedMoE moe_mlp_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};
  const v32_sp::DeepseekV32SPContext* sequence_parallel_context_ = nullptr;
};

TORCH_MODULE(DeepseekV2DecoderLayer);
}  // namespace layer
}  // namespace janus
