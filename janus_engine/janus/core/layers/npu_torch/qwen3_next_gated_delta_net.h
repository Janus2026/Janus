#pragma once

#include <torch/torch.h>

#include "attention.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm_gated.h"

namespace janus {
namespace layer {

class Qwen3NextGatedDeltaNetImpl : public torch::nn::Module {
 public:
  Qwen3NextGatedDeltaNetImpl() = default;
  Qwen3NextGatedDeltaNetImpl(const ModelArgs& args,
                             const QuantArgs& quant_args,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options);

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

  void load_state_dict(const StateDict& state_dict);

 private:
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  process_qkvz_tensor(const torch::Tensor& qkvz);
  std::tuple<torch::Tensor, torch::Tensor> process_ba_tensor(
      const torch::Tensor& ba);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> process_mixed_qkv(
      torch::Tensor& mixed_qkv);

  torch::Tensor reshape_qkvz_with_pad(const AttentionMetadata& attn_metadata,
                                      const torch::Tensor& qkvz);
  torch::Tensor reshape_qkvz_unpad(const AttentionMetadata& attn_metadata,
                                   const torch::Tensor& padded_qkvz);

  int64_t num_k_heads_;
  int64_t num_v_heads_;
  int64_t num_kv_head_replicas_;
  int64_t head_k_dim_;
  int64_t head_v_dim_;
  int64_t k_size_;
  int64_t v_size_;
  int64_t tp_size_;
  int64_t rank_;
  int32_t conv_kernel_size_;

  ColumnParallelLinear qkvz_proj_{nullptr};
  ColumnParallelLinear ba_proj_{nullptr};
  ColumnParallelLinear conv1d_{nullptr};

  RowParallelLinear o_proj_{nullptr};

  RmsNormGated norm_{nullptr};
  DEFINE_WEIGHT(dt_bias);
  DEFINE_WEIGHT(A_log);
};
TORCH_MODULE(Qwen3NextGatedDeltaNet);

}  // namespace layer
}  // namespace janus
