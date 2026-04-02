#pragma once

#include "common/metrics.h"
#include "forward_params.h"
#include "runtime/mtp_worker_impl.h"

namespace janus {

class Eagle3WorkerImpl : public MTPWorkerImpl {
 public:
  Eagle3WorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options);

  ~Eagle3WorkerImpl() override = default;

  // Override init_model to load hot_token_id_ for EAGLE-3
  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

  // EAGLE-3 draft input_embedding is 3 * target_hidden_size
  int64_t get_embedding_placeholder_size() override;

  // Override validate to handle EAGLE-3 specific token ID mapping
  SampleOutput validate(const SamplingParameters& sampling_params,
                        const std::vector<ForwardOutput>& draft_outputs,
                        const ForwardOutput& target_output) override;

  // Get hot_token_id for draft-to-target token mapping
  torch::Tensor get_hot_token_id() const { return hot_token_id_; }

 protected:
  // EAGLE-3 specific draft output post-processing during decode:
  // selected prob extraction + draft->target token id mapping.
  void process_draft_sample_output(SampleOutput& sample_output) override;

  // EAGLE-3 specific: hot_token_id for draft-to-target token mapping
  // hot_token_id = d2t + arange(d2t.size(0))
  torch::Tensor hot_token_id_;
};

}  // namespace janus
