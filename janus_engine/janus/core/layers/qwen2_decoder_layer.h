#pragma once

#include <torch/torch.h>

#include <optional>
#include <tuple>

#include "common/dense_mlp.h"
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

class Qwen2DecoderLayerImpl : public torch::nn::Module {
 public:
  explicit Qwen2DecoderLayerImpl(const ModelContext& context);

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  Qwen2Attention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};

  ParallelArgs parallel_args_;

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> apply_norm(
      RMSNorm& norm,
      torch::Tensor& input,
      std::optional<torch::Tensor>& residual,
      const std::optional<torch::Tensor>& fp8_scale);
};
TORCH_MODULE(Qwen2DecoderLayer);

}  // namespace layer
}  // namespace janus
