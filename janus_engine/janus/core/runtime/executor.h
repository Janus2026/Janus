#pragma once

#include <torch/torch.h>

#include "framework/batch/batch.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "framework/model/model_output.h"
#include "runtime/executor_impl.h"
#include "runtime/options.h"

namespace janus {

class Executor final {
 public:
  Executor(CausalLM* model,
           const ModelArgs& args,
           const torch::Device& device,
           const runtime::Options& options);

  virtual ~Executor() = default;

  ForwardInput prepare_inputs(Batch& batch);

  // tokens: vector size is dp_size, each element is [num_tokens/dp_size]
  // positions: vector size is dp_size, each element is [num_tokens/dp_size]
  // token pos in the sequence returns: ModelOutput
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params);

 private:
  std::unique_ptr<ExecutorImpl> impl_;
};

}  // namespace janus
