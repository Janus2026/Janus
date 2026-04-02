#pragma once
#include <torch/torch.h>

namespace janus {
namespace layer {

class AttentionMask : public torch::nn::Module {
 public:
  AttentionMask() = default;

  explicit AttentionMask(at::Device device,
                         torch::Dtype dtype,
                         float mask_value = -9984);

  torch::Tensor get_decode_attn_mask(torch::Tensor input_lengths,
                                     int64_t max_s,
                                     torch::Dtype dtype,
                                     torch::Device device);

  torch::Tensor get_attn_mask(int64_t max_s,
                              torch::Dtype dtype,
                              torch::Device device);

  torch::Tensor gen_free_mask(int32_t q_len,
                              torch::Dtype dtype,
                              torch::Device device);

  torch::Tensor gen_append_mask(int32_t q_len,
                                int32_t kv_len,
                                int32_t max_kv_len,
                                torch::Dtype dtype,
                                torch::Device device);

 private:
  void update_attn_cache(torch::Dtype dtype,
                         torch::Device device,
                         int64_t seqlen);

  int seq_len_cached_;
  float mask_value_;
  at::Tensor atten_mask_cache_;
};

}  // namespace layer
}  // namespace janus
