#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "executor_impl_factory.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "runtime/executor_impl.h"
#include "runtime/options.h"

namespace janus {

class BaseExecutorImpl : public ExecutorImpl {
 public:
  BaseExecutorImpl(CausalLM* model,
                   const ModelArgs& args,
                   const torch::Device& device,
                   const runtime::Options& options);

  ~BaseExecutorImpl() override = default;

  ForwardInput prepare_inputs(Batch& batch) override;

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params) override;

 private:
  // not own
  CausalLM* model_;

  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;
};
REGISTER_EXECUTOR("llm", BaseExecutorImpl);

using RecExecutorImpl = BaseExecutorImpl;
REGISTER_EXECUTOR("rec", RecExecutorImpl);
}  // namespace janus
