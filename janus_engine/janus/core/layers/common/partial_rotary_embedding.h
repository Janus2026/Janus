#pragma once

#include <torch/torch.h>
#include <torch/types.h>

#include <memory>

namespace janus {
namespace layer {
class PartialRotaryEmbeddingImpl : public torch::nn::Module {
 public:
  PartialRotaryEmbeddingImpl(int64_t rotary_dim,
                             int64_t max_position_embeddings,
                             int64_t rope_theta,
                             int64_t head_size,
                             bool is_neox_style,
                             bool interleaved,
                             const torch::TensorOptions& options);

  void forward(const torch::Tensor& positions,
               torch::Tensor& q,
               torch::Tensor& k);

  torch::Tensor get_cos_sin_cache() { return cos_sin_cache_; }

 private:
  int64_t head_size_;
  int64_t rotary_dim_;
  bool is_neox_style_;
  bool interleaved_;
  torch::Tensor cos_sin_cache_;
};
TORCH_MODULE(PartialRotaryEmbedding);

}  // namespace layer
}  // namespace janus
