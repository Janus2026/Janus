#pragma once

#include <torch/torch.h>

#include "common/dense_mlp.h"
#if defined(USE_MLU)
#include "layers/mlu/fused_moe.h"
#elif defined(USE_ILU)
#include "layers/ilu/fused_moe.h"
#else
#include "layers/common/fused_moe.h"
#endif
#include "common/qwen2_attention.h"
#include "common/rms_norm.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/state_dict/state_dict.h"

namespace janus {
namespace layer {

class Qwen3MoeDecoderLayerImpl : public torch::nn::Module {
 public:
  explicit Qwen3MoeDecoderLayerImpl(const ModelContext& context,
                                    int32_t layer_id);

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  torch::Tensor run_moe(torch::Tensor x, const ModelInputParams& input_params);

  Qwen2Attention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  FusedMoE moe_mlp_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};
  ParallelArgs parallel_args_;
  bool enable_deep_ep_ = false;
};
TORCH_MODULE(Qwen3MoeDecoderLayer);

}  // namespace layer
}  // namespace janus
