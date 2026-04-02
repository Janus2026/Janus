#pragma once

#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/macros.h"
#include "common/options.h"
#include "loadbalance_policy.h"

namespace janus_service {

// Job representation for the scheduling algorithm
struct SchedulingJob {
  std::shared_ptr<Request> request;
  int64_t processing_time_ms;  // p_j: estimated prefill time
  int64_t deadline_ms;         // d_j: arrival_time + ttft_slo
};

// Machine (instance) representation for the scheduling algorithm
struct MachineInfo {
  std::string instance_name;
  int64_t availability_time_ms;  // T_i: estimated time until free
};

// Internal pending entry: pairs a request with a promise for synchronous
// blocking in select_instances_pair().
struct PendingEntry {
  std::shared_ptr<Request> request;
  std::shared_ptr<std::promise<bool>> result_promise;
  int slo_expansion_count = 0;
};

class LstImhPolicy final : public LoadBalancePolicy {
 public:
  LstImhPolicy(const Options& options,
                std::shared_ptr<InstanceMgr> instance_mgr);
  ~LstImhPolicy() override;

  bool select_instances_pair(std::shared_ptr<Request> request) override;
  void on_prefill_done(const std::string& instance_name) override;
  void shutdown() override;

 private:
  DISALLOW_COPY_AND_ASSIGN(LstImhPolicy);

  // Coordinator thread main loop: collects pending requests per model,
  // runs LST-IMH scheduling, and fulfills promises.
  void dispatch_coordinator();

  // Signal the coordinator to wake up for re-evaluation.
  void signal_dispatch();

  // Run LST-IMH algorithm: assign jobs to machines maximizing on-time
  // completions.  Returns: instance_name -> vector of assigned jobs (sorted by
  // EDD).
  std::unordered_map<std::string, std::vector<SchedulingJob>> run_lst_imh(
      std::vector<SchedulingJob>& jobs,
      std::vector<MachineInfo>& machines);

  // Moore-Hodgson algorithm for single machine: maximize on-time jobs.
  // Returns jobs that can be completed on-time, sorted by EDD.
  std::vector<SchedulingJob> moore_hodgson(
      const std::vector<SchedulingJob>& jobs,
      int64_t T);

 private:
  Options options_;
  std::atomic<bool> exited_{false};

  // Per-model pending queues (requests waiting for coordinator dispatch).
  std::mutex pending_mutex_;
  std::unordered_map<std::string, std::vector<std::shared_ptr<PendingEntry>>>
      pending_queues_;

  // Coordinator thread synchronization.
  std::mutex dispatch_mutex_;
  std::condition_variable dispatch_cv_;
  bool dispatch_signal_ = false;

  std::unique_ptr<std::thread> coordinator_thread_;

  // MixPD decode instance round-robin index
  size_t decode_rr_idx_ = 0;
};

}  // namespace janus_service
