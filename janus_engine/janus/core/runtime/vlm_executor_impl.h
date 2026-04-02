#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "executor_impl_factory.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/causal_vlm.h"
#include "framework/model/model_input_params.h"
#include "framework/model/model_output.h"
#include "runtime/executor_impl.h"
#include "runtime/options.h"

namespace janus {

class VlmExecutorImpl : public ExecutorImpl {
 public:
  VlmExecutorImpl(CausalLM* model,
                  const ModelArgs& args,
                  const torch::Device& device,
                  const runtime::Options& options);

  ~VlmExecutorImpl() override = default;

  ForwardInput prepare_inputs(Batch& batch) override;

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params) override;

  virtual MMDict encode(const ModelInputParams& params);

 private:
  // not own
  CausalVLM* model_;
  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;
  std::unique_ptr<ExecutorImpl> llm_executor_;
};
// Q: backend device ?
REGISTER_EXECUTOR("vlm", VlmExecutorImpl);
}  // namespace janus
