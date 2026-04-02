#pragma once

#include "scheduler/chunked_prefill_scheduler.h"

namespace janus {
class PrefillOnlyScheduler final : public ContinuousScheduler {
 public:
  PrefillOnlyScheduler(Engine* engine, const Options& options);
  virtual ~PrefillOnlyScheduler();

 private:
  // build a batch of requests from the priority queue
  virtual std::vector<Batch> prepare_batch() override;
  void handle_prefill_requests(
      double& latency_budget,
      double& estimate_latency,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      RequestPriorityQueue& waiting_priority_queue,
      size_t& num_online_prefill_preempt_offline_requests,
      std::vector<std::shared_ptr<Request>>& finished_requests);
  void handle_last_step_prefill_requests(
      double& latency_budget,
      double& estimate_latency,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      std::vector<std::shared_ptr<Request>>& running_requests,
      size_t& num_online_prefill_preempt_offline_requests,
      std::vector<std::shared_ptr<Request>>& finished_requests);
};
}  // namespace janus