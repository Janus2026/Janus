#pragma once

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include "executor.h"
#include "forward_params.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "options.h"
#include "runtime/worker_impl.h"

namespace janus {

class LLMWorkerImpl : public WorkerImpl {
 public:
  LLMWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options);

  ~LLMWorkerImpl() override = default;

  // initialize model, cache manager. blocking call
  bool init_model(ModelContext& context) override;

  std::optional<ForwardOutput> step(const ForwardInput& input) override;

  std::optional<ForwardOutput> step_internal(const ForwardInput& input);

#if defined(USE_NPU)
  layer::NpuLmHead get_npu_lm_head() { return model_->get_npu_lm_head(); };

  void set_npu_lm_head(layer::NpuLmHead& head) {
    model_->set_npu_lm_head(head);
  };

  layer::NpuWordEmbedding get_npu_word_embedding() {
    return model_->get_npu_word_embedding();
  };

  void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) {
    model_->set_npu_word_embedding(embedding);
  };

#endif
  layer::LmHead get_lm_head() { return model_->get_lm_head(); };

  void set_lm_head(layer::LmHead& head) { model_->set_lm_head(head); };

  layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  };

  void set_word_embedding(layer::WordEmbedding& embedding) {
    model_->set_word_embedding(embedding);
  };

 protected:
  std::unique_ptr<BeamSearcher> beam_searcher_;
};

}  // namespace janus
