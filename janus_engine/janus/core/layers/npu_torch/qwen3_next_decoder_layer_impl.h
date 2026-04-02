#pragma once

#include <torch/torch.h>

#include <functional>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/dense_mlp.h"
#include "layers/common/qwen3_next_rms_norm.h"
#include "layers/npu_torch/fused_moe.h"
#include "layers/npu_torch/qwen3_next_attention.h"
#include "layers/npu_torch/qwen3_next_gated_delta_net.h"

namespace janus {
namespace layer {

class Qwen3NextDecoderLayerImpl : public torch::nn::Module {
 public:
  explicit Qwen3NextDecoderLayerImpl(const ModelContext& context,
                                     int32_t layer_id);

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  Qwen3NextAttention attention_{nullptr};
  Qwen3NextGatedDeltaNet linear_attention_{nullptr};

  DenseMLP mlp_{nullptr};
  FusedMoE moe_mlp_{nullptr};

  Qwen3NextRMSNorm input_norm_{nullptr};
  Qwen3NextRMSNorm post_norm_{nullptr};
};
TORCH_MODULE(Qwen3NextDecoderLayer);

}  // namespace layer
}  // namespace janus
