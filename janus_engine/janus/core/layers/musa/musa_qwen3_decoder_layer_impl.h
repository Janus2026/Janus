#pragma once

#include <torch/torch.h>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/attention_metadata.h"
#include "musa_layer_base.h"

namespace janus {
namespace layer {
class Qwen3DecoderLayerImpl : public torch::nn::Module {
 public:
  explicit Qwen3DecoderLayerImpl(const ModelContext& context);

  ~Qwen3DecoderLayerImpl() {};

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  std::vector<std::shared_ptr<MUSALayerBaseImpl>> layers_;
  // ParallelArgs parallel_args_;
};
TORCH_MODULE(Qwen3DecoderLayer);

}  // namespace layer
}  // namespace janus