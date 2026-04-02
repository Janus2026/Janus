#pragma once

#include <torch/torch.h>

#include "attention.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/linear.h"
#include "layers/common/partial_rotary_embedding.h"
#include "layers/common/qwen3_next_rms_norm.h"

namespace janus {
namespace layer {

class Qwen3NextAttentionImpl : public torch::nn::Module {
 public:
  Qwen3NextAttentionImpl() = default;
  Qwen3NextAttentionImpl(const ModelArgs& args,
                         const QuantArgs& quant_args,
                         const ParallelArgs& parallel_args,
                         const torch::TensorOptions& options,
                         int32_t layer_id);

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache);

  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t num_heads_;
  int64_t num_kv_heads_;
  int64_t num_kv_head_replicas_;
  int64_t head_dim_;
  int64_t q_size_;
  int64_t kv_size_;
  float scaling_;
  bool attn_output_gate_;
  int32_t layer_id_;
  int32_t rank_;

  QKVParallelLinear qkv_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};

  Qwen3NextRMSNorm q_norm_{nullptr};
  Qwen3NextRMSNorm k_norm_{nullptr};

  Attention attn_{nullptr};
  PartialRotaryEmbedding rotary_emb_{nullptr};
};
TORCH_MODULE(Qwen3NextAttention);

}  // namespace layer
}  // namespace janus
