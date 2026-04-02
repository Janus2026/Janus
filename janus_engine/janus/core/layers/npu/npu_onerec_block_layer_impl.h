#pragma once

#include <torch/torch.h>

#include <cstdint>

#include "framework/model_context.h"
#include "layers/onerec_block_layer.h"

namespace janus {
namespace layer {

class NpuOneRecBlockLayerImpl final : public OneRecBlockLayer {
 public:
  explicit NpuOneRecBlockLayerImpl(const ModelContext& context,
                                   bool is_decoder = false,
                                   int32_t layer_id = 0);

  torch::Tensor forward(torch::Tensor& hidden_states,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        torch::Tensor* encoder_output = nullptr,
                        int32_t node_id = 0,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr) override;

 private:
  const torch::Device device_;
  bool is_decoder_ = false;
  int32_t layer_id_ = 0;
};

}  // namespace layer
}  // namespace janus
