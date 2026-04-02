#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "framework/batch/batch.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "framework/model/model_output.h"
#include "options.h"

namespace janus {

class ExecutorImpl {
 public:
  virtual ~ExecutorImpl() = default;

  virtual ForwardInput prepare_inputs(Batch& batch) = 0;

  // tokens: vector size is dp_size, each element is [num_tokens/dp_size]
  // positions: vector size is dp_size, each element is [num_tokens/dp_size]
  // token pos in the sequence returns: ModelOutput
  virtual ModelOutput run(const torch::Tensor& tokens,
                          const torch::Tensor& positions,
                          std::vector<KVCache>& kv_caches,
                          const ModelInputParams& params) = 0;
};

}  // namespace janus
