#pragma once

#include <torch/torch.h>

#include "attention.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/rms_norm.h"
#include "linear.h"
#include "rotary_embedding.h"

namespace janus {
namespace layer {

class Qwen2AttentionImpl : public torch::nn::Module {
 public:
  Qwen2AttentionImpl() = default;
  Qwen2AttentionImpl(const ModelContext& context);

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache);

  void load_state_dict(const StateDict& state_dict);

  // Get FP8 input scale from qkv_proj for fused RMSNorm+FP8 quantization
  std::optional<torch::Tensor> get_fp8_input_scale() const;

 private:
  int64_t num_heads_;
  int64_t num_kv_heads_;
  int64_t num_kv_head_replicas_;
  int64_t head_dim_;
  int64_t q_size_;
  int64_t kv_size_;
  float scaling_;
  bool is_qwen3_style_;
  bool can_use_fused_qk_norm_rope_;

  QKVParallelLinear qkv_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  RMSNorm q_norm_{nullptr};
  RMSNorm k_norm_{nullptr};
  Attention attn_{nullptr};
  MRotaryEmbedding rotary_emb_{nullptr};
};
TORCH_MODULE(Qwen2Attention);

}  // namespace layer
}  // namespace janus
