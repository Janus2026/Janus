#pragma once

#include <map>
#include <vector>

#include "base_loader.h"

namespace janus {
namespace layer {

class Qwen3VisionEncoderLoader : public BaseLoader {
 public:
  Qwen3VisionEncoderLoader(uint64_t weight_count, const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  void merge_loaded_weights() override;

 private:
  void get_weights_col_packed_qkv();

 protected:
  std::string model_name_;
  at::Tensor cu_seqlen_;
  at::Tensor at_placeholder_;
  std::vector<torch::Tensor> qkv_weight;
  std::vector<torch::Tensor> qkv_bias;
  int device_id_;
  int encode_param_rank;
  int encode_param_worldSize;
};

}  // namespace layer
}  // namespace janus