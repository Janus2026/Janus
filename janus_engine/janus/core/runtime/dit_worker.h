#pragma once

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <thread>

#include "dit_executor.h"
#include "framework/dit_model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "options.h"
#include "platform/device.h"
#include "util/threadpool.h"

namespace janus {

class DiTWorker {
 public:
  DiTWorker(const ParallelArgs& parallel_args,
            const torch::Device& device,
            const runtime::Options& options);

  ~DiTWorker() = default;

  // initialize model, cache manager. blocking call
  bool init_model(const std::string& model_weights_path);

  folly::SemiFuture<bool> init_model_async(
      const std::string& model_weights_path);

  std::optional<DiTForwardOutput> step(const DiTForwardInput& inputs);

  folly::SemiFuture<std::optional<DiTForwardOutput>> step_async(
      const DiTForwardInput& inputs);

  void process_group_test();

  folly::SemiFuture<folly::Unit> process_group_test_async();

  // prepare input for execution
  DiTForwardInput prepare_inputs(DiTBatch& batch);

  int64_t get_active_activation_memory();

 private:
  runtime::Options options_;

  std::unique_ptr<DiTModel> dit_model_;

  std::unique_ptr<DiTExecutor> dit_model_executor_;

  Device device_;

  torch::ScalarType dtype_;

  // model context, includes model args, parallel args and date type etc.
  mutable DiTModelContext context_;

  ParallelArgs parallel_args_;

  ThreadPool threadpool_;
};

}  // namespace janus
