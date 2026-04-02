#pragma once

#include <map>
#include <vector>

#include "base_loader.h"

namespace janus {
namespace layer {

class Qwen2dot5VisionEncoderLoader : public BaseLoader {
 public:
  Qwen2dot5VisionEncoderLoader(uint64_t weight_count,
                               const ModelContext& context,
                               int64_t numAttentionHeadsPerRank);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  void merge_loaded_weights() override;

 private:
  void get_weights_col_packed_qkv();
  void pad_qkv_weights();
  void pad_mlp_weights();
  torch::Tensor pad_tensor(const torch::Tensor& tensor,
                           int64_t target_shape,
                           int64_t dim = 0);

 protected:
  std::string model_name_;
  torch::Tensor cu_seqlen_;
  torch::Tensor at_placeholder_;
  std::vector<torch::Tensor> qkv_weight;
  std::vector<torch::Tensor> qkv_bias;
  int device_id_;
  int encode_param_rank;
  int encode_param_worldSize;
  int64_t encode_param_numAttentionHeadsPerRank;
};

}  // namespace layer
}  // namespace janus