#pragma once

#include <torch/torch.h>

#include <atomic>
#include <cstdint>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"

#if defined(USE_NPU)
#include <acl/acl.h>
#else
using aclrtEvent = void*;
#endif

namespace janus {
namespace layer {

class OneRecBlockLayer : public torch::nn::Module {
 public:
  virtual ~OneRecBlockLayer() = default;

  virtual torch::Tensor forward(torch::Tensor& hidden_states,
                                torch::Tensor& attn_mask,
                                KVCache& kv_cache,
                                ModelInputParams& input_params,
                                torch::Tensor* encoder_output = nullptr,
                                int32_t node_id = 0,
                                aclrtEvent* event = nullptr,
                                std::atomic<bool>* event_flag = nullptr) = 0;
};

}  // namespace layer
}  // namespace janus
