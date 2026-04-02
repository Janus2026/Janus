#pragma once

#include <torch/torch.h>

#include <functional>

#include "common/dense_mlp.h"
#include "common/qwen2_vision_attention.h"
#include "common/rms_norm.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"

namespace janus {
namespace layer {

class Qwen2_5_VisionLayerImpl : public torch::nn::Module {
 public:
  Qwen2_5_VisionLayerImpl(const ModelContext& context,
                          bool is_qwen3_style = false);

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& m_cos_pos,
                        torch::Tensor& m_sin_pos,
                        torch::Tensor& cu_seq_len,
                        std::vector<int32_t>& cu_seq_len_vec,
                        ModelInputParams& input_params,
                        int node_id);

 protected:
  Qwen2VisionAttention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  RMSNorm norm1_{nullptr};
  RMSNorm norm2_{nullptr};
};
TORCH_MODULE(Qwen2_5_VisionLayer);

}  // namespace layer
}  // namespace janus
