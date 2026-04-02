#include "lst_imh_policy.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <unordered_set>

#include "common/types.h"

namespace janus_service {

LstImhPolicy::LstImhPolicy(const Options& options,
                             std::shared_ptr<InstanceMgr> instance_mgr)
    : options_(options), LoadBalancePolicy(instance_mgr) {
  coordinator_thread_ =
      std::make_unique<std::thread>(&LstImhPolicy::dispatch_coordinator, this);
  // LOG(INFO) << "LST-IMH scheduling mode enabled (as LoadBalancePolicy).";
}

LstImhPolicy::~LstImhPolicy() { shutdown(); }

// ---------------------------------------------------------------------------
// select_instances_pair — called by process_request_queue threads
//
// Blocks until the coordinator assigns an instance (returns true) or the
// request's TTFT SLO expires (returns false).
// ---------------------------------------------------------------------------
bool LstImhPolicy::select_instances_pair(std::shared_ptr<Request> request) {
  // Compute estimated processing time if not already set.
  if (!request->token_ids.empty() &&
      request->estimated_processing_time_ms == 0) {
    request->estimated_processing_time_ms = static_cast<int64_t>(
        instance_mgr_->predict_ttft_any_instance(request->model,
                                                  request->token_ids.size()));
  }

  // Create a promise/future pair for synchronous blocking.
  auto promise_ptr = std::make_shared<std::promise<bool>>();
  std::future<bool> fut = promise_ptr->get_future();

  auto entry = std::make_shared<PendingEntry>();
  entry->request = request;
  entry->result_promise = promise_ptr;

  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_queues_[request->model].push_back(entry);
  }

  // Wake the coordinator so it can schedule this request.
  signal_dispatch();

  // Block until the coordinator fulfills the promise or hard timeout expires.
  // The wait covers the original SLO plus all possible SLO expansions, so the
  // coordinator has time to re-enqueue the request with relaxed deadlines.
  int64_t base_ms = request->ttft_slo_ms > 0 ? request->ttft_slo_ms : 30000;
  double max_factor =
      std::pow(options_.slo_penalty_factor(), options_.max_slo_expansions());
  int64_t wait_ms = static_cast<int64_t>(base_ms * max_factor) + 5000;
  auto status = fut.wait_for(std::chrono::milliseconds(wait_ms));

  if (status == std::future_status::timeout) {
    // Timed out — remove from pending queue.
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto& q = pending_queues_[request->model];
      q.erase(std::remove_if(q.begin(), q.end(),
                              [&request](const std::shared_ptr<PendingEntry>& e) {
                                return e->request->service_request_id ==
                                       request->service_request_id;
                              }),
              q.end());
    }

    LOG(WARNING) << "[LST-IMH] Request " << request->service_request_id
                 << " timed out waiting for instance assignment (model="
                 << request->model << ")";
    if (request->timeout_callback) {
      request->timeout_callback();
    }
    return false;
  }

  return fut.get();
}

// ---------------------------------------------------------------------------
// on_prefill_done — called when a prefill finishes on an instance
// ---------------------------------------------------------------------------
void LstImhPolicy::on_prefill_done(const std::string& instance_name) {
  signal_dispatch();
}

// ---------------------------------------------------------------------------
// shutdown — unblock all threads and stop the coordinator
// ---------------------------------------------------------------------------
void LstImhPolicy::shutdown() {
  bool was_running = !exited_.exchange(true);
  if (!was_running) return;

  // Wake coordinator thread.
  {
    std::lock_guard<std::mutex> lock(dispatch_mutex_);
    dispatch_signal_ = true;
  }
  dispatch_cv_.notify_one();

  if (coordinator_thread_ && coordinator_thread_->joinable()) {
    coordinator_thread_->join();
  }

  // Fulfill all pending promises with false so blocked threads can exit.
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    for (auto& [model, queue] : pending_queues_) {
      for (auto& entry : queue) {
        try {
          entry->result_promise->set_value(false);
        } catch (const std::future_error&) {
          // Promise already satisfied — ignore.
        }
      }
      queue.clear();
    }
    pending_queues_.clear();
  }
}

// ---------------------------------------------------------------------------
// signal_dispatch — wake the coordinator
// ---------------------------------------------------------------------------
void LstImhPolicy::signal_dispatch() {
  {
    std::lock_guard<std::mutex> lock(dispatch_mutex_);
    dispatch_signal_ = true;
  }
  dispatch_cv_.notify_one();
}

// ---------------------------------------------------------------------------
// dispatch_coordinator — main loop (runs in dedicated thread)
//
// Migrated from Scheduler::dispatch_coordinator with multi-model support.
// ---------------------------------------------------------------------------
void LstImhPolicy::dispatch_coordinator() {
  const int64_t pre_pull_ms =
      static_cast<int64_t>(options_.lst_imh_pre_pull_ms());
  // LOG(INFO) << "[LST-IMH] dispatch_coordinator started, pre_pull_ms="
  //           << pre_pull_ms;

  while (!exited_) {
    // ---- Phase 1: Compute next wake time from EPDT across all models ----
    int64_t now_ms = absl::ToUnixMillis(absl::Now());
    int64_t next_wake_ms = std::numeric_limits<int64_t>::max();

    std::vector<std::string> active_models;
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      for (const auto& [model, queue] : pending_queues_) {
        if (!queue.empty()) {
          active_models.push_back(model);
        }
      }
    }

    for (const auto& model : active_models) {
      auto instances = instance_mgr_->get_awake_prefill_instances(model);
      for (const auto& inst : instances) {
        int64_t epdt =
            instance_mgr_->get_estimated_prefill_done_time(inst);
        if (epdt > now_ms) {
          int64_t wake_at = epdt - pre_pull_ms;
          if (wake_at > now_ms) {
            next_wake_ms = std::min(next_wake_ms, wake_at);
          }
        }
      }
    }

    // ---- Phase 2: Wait for signal or timer ----
    {
      std::unique_lock<std::mutex> lock(dispatch_mutex_);
      if (next_wake_ms == std::numeric_limits<int64_t>::max()) {
        dispatch_cv_.wait(lock, [this] {
          return dispatch_signal_ || exited_.load();
        });
      } else {
        auto wait_ms_val =
            std::max(int64_t(0), next_wake_ms - now_ms);
        dispatch_cv_.wait_for(lock,
                              std::chrono::milliseconds(wait_ms_val),
                              [this] {
                                return dispatch_signal_ || exited_.load();
                              });
      }
      dispatch_signal_ = false;
    }

    if (exited_) break;

    now_ms = absl::ToUnixMillis(absl::Now());

    // Refresh active models list (may have changed while waiting).
    active_models.clear();
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      for (const auto& [model, queue] : pending_queues_) {
        if (!queue.empty()) {
          active_models.push_back(model);
        }
      }
    }

    // ---- Phase 3: For each model, run LST-IMH and dispatch ----
    for (const auto& model : active_models) {
      auto awake_instances = instance_mgr_->get_awake_prefill_instances(model);
      if (awake_instances.empty()) continue;

      // Build machine list with real-time availability estimates using EPDT.
      std::vector<MachineInfo> machines;
      std::vector<std::string> ready_instances;

      for (const auto& inst : awake_instances) {
        int64_t epdt =
            instance_mgr_->get_estimated_prefill_done_time(inst);
        int64_t T_i = std::max(int64_t(0), epdt - now_ms);
        machines.push_back({inst, T_i});
        if (T_i <= pre_pull_ms) {
          ready_instances.push_back(inst);
        }
      }

      if (ready_instances.empty()) continue;

      // Purge expired requests or expand their SLO for re-scheduling.
      std::vector<SchedulingJob> jobs;
      {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        auto& queue = pending_queues_[model];
        auto it = queue.begin();
        while (it != queue.end()) {
          auto& entry = *it;
          int64_t deadline =
              entry->request->arrival_time_ms + entry->request->ttft_slo_ms;
          if (deadline <= now_ms) {
            if (entry->slo_expansion_count <
                options_.max_slo_expansions()) {
              // SLO expired — expand and re-enqueue with lower priority.
              entry->request->ttft_slo_ms = static_cast<int64_t>(
                  entry->request->ttft_slo_ms * options_.slo_penalty_factor());
              entry->slo_expansion_count++;
              int64_t new_deadline =
                  entry->request->arrival_time_ms + entry->request->ttft_slo_ms;
              // LOG(INFO) << "[LST-IMH] Request "
              //           << entry->request->service_request_id
              //           << " SLO expanded (x" << entry->slo_expansion_count
              //           << ", new_slo=" << entry->request->ttft_slo_ms
              //           << "ms, new_deadline=" << new_deadline << ")";
              jobs.push_back({entry->request,
                              entry->request->estimated_processing_time_ms,
                              new_deadline});
              ++it;
            } else {
              // Max expansions reached — truly discard.
              LOG(WARNING) << "[LST-IMH] Request "
                           << entry->request->service_request_id
                           << " expired after " << entry->slo_expansion_count
                           << " SLO expansions, discarding.";
              try {
                entry->result_promise->set_value(false);
              } catch (const std::future_error&) {}
              if (entry->request->timeout_callback) {
                entry->request->timeout_callback();
              }
              it = queue.erase(it);
            }
          } else {
            jobs.push_back({entry->request,
                            entry->request->estimated_processing_time_ms,
                            deadline});
            ++it;
          }
        }
      }

      if (jobs.empty()) continue;

      // Run LST-IMH algorithm.
      auto assignment = run_lst_imh(jobs, machines);

      // Dispatch to each ready instance.
      for (const auto& instance : ready_instances) {
        auto assign_it = assignment.find(instance);
        if (assign_it == assignment.end() || assign_it->second.empty()) {
          continue;
        }

        // Take first job (already in EDD order from Moore-Hodgson).
        auto& job = assign_it->second[0];
        auto request = job.request;

        // Set routing.
        request->routing.prefill_name = instance;

        auto tag = instance_mgr_->get_instance_tag(instance);
        if (tag == InstanceTag::PREFILL) {
          // PREFILL-tagged: round-robin select from decode instances
          auto decode_instances = instance_mgr_->get_awake_decode_instances(model);
          if (!decode_instances.empty()) {
            request->routing.decode_name =
                decode_instances[decode_rr_idx_ % decode_instances.size()];
            ++decode_rr_idx_;
          } else {
            // Fallback: self-decode if no decode instances available
            request->routing.decode_name = instance;
          }
        } else {
          // NORMAL or NONE tagged: self-decode
          request->routing.decode_name = instance;
        }
        request->estimated_ttft = job.processing_time_ms;

        // Update request metrics eagerly (prevents double-allocation within
        // same coordinator cycle).
        if (!request->prompt.empty()) {
          instance_mgr_->update_request_metrics(request,
                                                RequestAction::SCHEDULE);
          request->metrics_already_updated = true;
        }

        // Remove from pending queue and fulfill promise.
        {
          std::lock_guard<std::mutex> lock(pending_mutex_);
          auto& queue = pending_queues_[model];
          for (auto it = queue.begin(); it != queue.end(); ++it) {
            if ((*it)->request->service_request_id ==
                request->service_request_id) {
              try {
                (*it)->result_promise->set_value(true);
              } catch (const std::future_error&) {}
              queue.erase(it);
              break;
            }
          }
        }
      }
    }  // for each model
  }    // while (!exited_)
}

// ---------------------------------------------------------------------------
// moore_hodgson — single-machine deadline scheduling (migrated from scheduler)
// ---------------------------------------------------------------------------
std::vector<SchedulingJob> LstImhPolicy::moore_hodgson(
    const std::vector<SchedulingJob>& jobs, int64_t T) {
  // Sort jobs by EDD (earliest due date first).
  std::vector<SchedulingJob> sorted_jobs = jobs;
  std::sort(sorted_jobs.begin(), sorted_jobs.end(),
            [](const SchedulingJob& a, const SchedulingJob& b) {
              return a.deadline_ms < b.deadline_ms;
            });

  // on_time_indices tracks which jobs (by index in sorted_jobs) are on-time.
  std::vector<size_t> on_time_indices;
  // Track removed indices for lazy heap deletion.
  std::unordered_set<size_t> removed_indices;

  // Max-heap: (processing_time, index_in_sorted_jobs)
  auto cmp = [](const std::pair<int64_t, size_t>& a,
                const std::pair<int64_t, size_t>& b) {
    return a.first < b.first;
  };
  std::priority_queue<std::pair<int64_t, size_t>,
                      std::vector<std::pair<int64_t, size_t>>,
                      decltype(cmp)>
      max_heap(cmp);

  int64_t current_time = T;

  for (size_t i = 0; i < sorted_jobs.size(); ++i) {
    on_time_indices.push_back(i);
    max_heap.push({sorted_jobs[i].processing_time_ms, i});
    current_time += sorted_jobs[i].processing_time_ms;

    if (current_time > sorted_jobs[i].deadline_ms) {
      // Skip stale heap entries (already removed in a previous iteration).
      while (!max_heap.empty() &&
             removed_indices.count(max_heap.top().second)) {
        max_heap.pop();
      }
      if (max_heap.empty()) break;

      // Remove job with largest processing time.
      auto [p_max, idx_max] = max_heap.top();
      max_heap.pop();

      // Mark as removed.
      removed_indices.insert(idx_max);

      // Remove idx_max from on_time_indices.
      on_time_indices.erase(
          std::remove(on_time_indices.begin(), on_time_indices.end(), idx_max),
          on_time_indices.end());

      current_time -= p_max;
    }
  }

  // Build result from on_time_indices (already in EDD order).
  std::vector<SchedulingJob> result;
  result.reserve(on_time_indices.size());
  for (size_t idx : on_time_indices) {
    result.push_back(sorted_jobs[idx]);
  }
  return result;
}

// ---------------------------------------------------------------------------
// run_lst_imh — multi-machine scheduling (migrated from scheduler)
// ---------------------------------------------------------------------------
std::unordered_map<std::string, std::vector<SchedulingJob>>
LstImhPolicy::run_lst_imh(std::vector<SchedulingJob>& jobs,
                           std::vector<MachineInfo>& machines) {
  // Sort machines by availability time ascending (earliest available first).
  // Randomize tie-breaking to avoid always favoring the same instance.
  static thread_local std::mt19937 rng(std::random_device{}());
  std::shuffle(machines.begin(), machines.end(), rng);
  std::stable_sort(machines.begin(), machines.end(),
                   [](const MachineInfo& a, const MachineInfo& b) {
                     return a.availability_time_ms < b.availability_time_ms;
                   });

  std::unordered_map<std::string, std::vector<SchedulingJob>> assignment;
  std::vector<SchedulingJob> remaining = jobs;

  for (const auto& machine : machines) {
    if (remaining.empty()) break;

    // Run Moore-Hodgson on remaining jobs for this machine.
    auto on_time = moore_hodgson(remaining, machine.availability_time_ms);

    if (!on_time.empty()) {
      // Only assign the most urgent job (first in EDD order) to this machine.
      // The dispatch loop dispatches at most 1 job per ready instance per
      // cycle, so assigning more would starve subsequent machines of work.
      assignment[machine.instance_name] = {on_time[0]};

      // Remove only the assigned job from remaining.
      const auto& assigned_id = on_time[0].request->service_request_id;
      remaining.erase(
          std::remove_if(
              remaining.begin(), remaining.end(),
              [&assigned_id](const SchedulingJob& j) {
                return j.request->service_request_id == assigned_id;
              }),
          remaining.end());
    }
  }

  return assignment;
}

}  // namespace janus_service
