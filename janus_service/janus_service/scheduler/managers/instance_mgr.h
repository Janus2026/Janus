#pragma once

#include <brpc/channel.h>

#include <condition_variable>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <chrono>

#include "common/macros.h"
#include "common/options.h"
#include "common/threadpool.h"
#include "common/time_predictor.h"
#include "common/types.h"
#include "request/request.h"
#include "scheduler/etcd_client/etcd_client.h"
#include "scheduler/managers/model_instance_mgr.h"
#include "scheduler/resource_model/resource_model.h"
#include "janus_rpc_service.pb.h"

namespace janus_service {

class InstanceMgr final {
 public:

  // Loaded from models_config_path JSON at startup; each pair is
  // {service_name, model_path}.  Populated by load_models_config() in init().
  std::vector<std::pair<std::string, std::string>> MODELS;

  static constexpr int kMaxWakeupTimeoutms = 10000;

 public:
  explicit InstanceMgr(const Options& options,
                       const std::shared_ptr<EtcdClient>& etcd_client,
                       const bool is_master_service);

  ~InstanceMgr();

  InstanceMetaInfo get_instance_info(const std::string& instance_name);

  bool get_next_instance_pair(const std::string& model_id, Routing* routing);

  std::vector<std::string> get_static_decode_list(
      const std::string& instance_name);

  std::vector<std::string> get_static_prefill_list(
      const std::string& instance_name);

  void get_load_metrics(LoadBalanceInfos* infos);

  std::shared_ptr<brpc::Channel> get_channel(const std::string& instance_name);

  void record_load_metrics_update(const std::string& instance_name,
                                  const proto::LoadMetrics& load_metrics);
  bool upload_load_metrics();

  // update the recent token latency metrics for the corresponding instance
  void update_latency_metrics(const std::string& instance_name,
                              const proto::LatencyMetrics& latency_metrics);

  // update request metrics under different actions
  void update_request_metrics(std::shared_ptr<Request> request,
                              RequestAction action);

  // select instances based on the SLO
  bool select_instance_pair_on_slo(std::shared_ptr<Request> request);

  // Get the estimated prefill done time (absolute ms since epoch) for an instance
  int64_t get_estimated_prefill_done_time(const std::string& instance_name);

  void set_as_master();

  void fork_master_and_sleep(const std::string& instance_name,
                             std::shared_ptr<brpc::Channel> channel);

  void on_heartbeat(const std::string& instance_name);

  void send_model_sleep(const std::string& instance_name,
                        const std::string& model_id);

  bool send_model_wakeup(const std::string& instance_name,
                         const std::string& model_id,
                         bool memory_increased_in_advance);

  void update_model_heat(const std::string& model_id,
                         int64_t token_count,
                         int64_t input_len);
  
  int32_t get_wakeup_count(const std::string& model_id);

  std::vector<std::string> get_awake_instances(const std::string& model_id);
  std::vector<std::string> get_awake_prefill_instances(const std::string& model_id);
  std::vector<std::string> get_awake_decode_instances(const std::string& model_id);
  int count_awake_models_on_instance(const std::string& instance_name);
  // Count models with WAKEUP or ALLOCATED state on an instance (includes in-flight wakeups)
  int count_active_models_on_instance(const std::string& instance_name);
  InstanceTag get_instance_tag(const std::string& instance_name) const;
  void set_instance_tag(const std::string& instance_name, InstanceTag tag);
  bool is_model_waking_up(const std::string& model_id);

  // Trigger auto-scaling. Computes GPU targets for elastic pool models
  // using resource_model, then wakes/sleeps instances asynchronously.
  // Budget = total_gpus - steady_needed_gpus.
  void dynamic_part_auto_scaling();

  // Non-blocking variant: try_lock on allocation_mutex_; if the lock is already
  // held (another scaling in progress), return false immediately without scaling.
  bool try_dynamic_part_auto_scaling();

  // Signal the auto-scaling thread and wait until the model has at least one
  // PREFILL and one DECODE instance in WAKEUP state (or timeout).
  // Used for cold elastic model first-request path.
  void request_cold_elastic_wakeup(const std::string& model_id);

  std::shared_ptr<ModelInstanceMgr> get_model_instance_mgr(const std::string& model_id);

  // --- Dual-pool scheduling ---

  // Assign a new model to a pool based on current gpu_target.
  // If gpu_target <= 1: steady pool (bin-packed). If > 1: elastic pool.
  // Blocks until wakeup completes for steady pool models.
  void assign_model_to_pool(const std::string& model_id);

  // Get current pool assignment for a model
  PoolType get_model_pool(const std::string& model_id);

  // Check if steady pool model needs upgrade to elastic. Returns true if upgraded.
  bool steady_part_check_upgrading(const std::string& model_id);

  // Periodic: sleep models with 0 heat, repack bins, release unused instances.
  void steady_part_auto_repacking();

  // Periodic: demote elastic models with sustained low demand (gpu_target<=1)
  // back to steady pool via find_or_create_steady_bin.
  void elastic_to_steady_demotion();

  // Route a request to the steady pool instance hosting the model.
  bool route_to_steady_instance(std::shared_ptr<Request> request);

  // Update XTensor info for an instance from heartbeat
  void update_xtensor_info(const std::string& instance_name,
                           const proto::XTensorHeartbeatInfo& xtensor_info);

  // Get XTensor info for an instance
  std::optional<InstanceXTensorInfo> get_instance_xtensor_info(
      const std::string& instance_name);

  // Find a suitable source instance for D2D weight transfer
  // Returns D2DWakeupInfo with source instance info, or empty if no suitable source found
  std::optional<D2DWakeupInfo> find_d2d_source(const std::string& model_id,
                                                const std::string& target_instance_name);

  // Check if an instance has valid xtensor info from heartbeat
  bool has_valid_xtensor_info(const std::string& instance_name);

  // Get model size in bytes (prefers xtensor info, fallback to specs)
  uint64_t get_model_size_bytes(const std::string& model_id);

  // Get instance free space in bytes from xtensor info
  uint64_t get_instance_free_bytes(const std::string& instance_name);

  // Check if instance has enough space to load a model
  bool has_enough_space_for_model(const std::string& instance_name,
                                   const std::string& model_id);

  // Predict TTFT for a specific instance (heterogeneous instances, for SLO_AWARE)
  double predict_ttft(const std::string& instance_name,
                      const std::string& model_id, int32_t token_count);

  // Predict TTFT using any available instance (homogeneous instances, for LST_IMH)
  double predict_ttft_any_instance(const std::string& model_id, int32_t token_count);

  // Block until model on instance has 0 inflight requests. Returns false on error.
  bool wait_for_model_drain(const std::string& instance_name,
                             const std::string& model_id);

  // Locally adjust free pages after allocation (before next heartbeat)
  void deduct_free_pages(const std::string& instance_name, uint64_t bytes);

 private:
  void init_model_memory_specs();
  void init_model_resource_coefficients();
  void load_gp_steady_models(const std::string& path);
  void load_gp_dynamic_models(const std::string& path);
  double get_model_memory_size(const std::string& model_id);

  // Load MODELS from the JSON file specified by options_.models_config_path().
  // Each JSON element must have "service" and "model_path" string fields.
  // Aborts with a fatal log if the path is empty or the file cannot be parsed.
  void load_models_config();

  // --- Dual-pool private helpers ---

  // Auto-scaling implementation. Must be called with allocation_mutex_ held.
  void dynamic_part_auto_scaling_impl();

  // 2D First Fit: find a bin for a model, or create a new one.
  // If all instances occupied, reclaims from elastic pool (blocking).
  // Must be called with allocation_mutex_ held.
  // When allow_reclaim is false, skip Phase 3 (elastic pool reclamation).
  std::string find_or_create_steady_bin(const std::string& model_id,
                                         const ResourceNeeds& needs,
                                         bool allow_reclaim = true);

  // Full FFD repack of all steady bins.
  // Returns list of (model_id, old_instance, new_instance) moves.
  // Must be called with allocation_mutex_ held.
  std::vector<std::tuple<std::string, std::string, std::string>>
      repack_steady_bins();

  // Remove a model from its steady bin, reclaiming resources.
  // Must be called with allocation_mutex_ held.
  // If removal causes imbalance ratio R(B) < 0.5, triggers per-bin repack:
  // all remaining models are evicted and re-inserted via min cos heuristic.
  // Returns (model_id, old_instance, new_instance) moves from repack.
  std::vector<std::tuple<std::string, std::string, std::string>>
      remove_model_from_steady_bin(const std::string& model_id);

  // Get resource needs for a model using its current traffic stats (GP 4D input).
  ResourceNeeds get_model_resource_needs(const std::string& model_id);

  // Compute GPU target using traffic stats (preferred over legacy heat-only interface).
  int32_t compute_gpu_target_for_model(const std::string& model_id);

  // Check if instance is in the steady pool
  bool is_steady_pool_instance(const std::string& instance_name);

  // Get number of GPUs reserved by the steady pool
  int32_t steady_needed_gpus();

  // Compute elastic pool gpu_target from 3-second average token rate.
  // Uses linear regression on empirical capacity data to find the minimum
  // instance count whose max throughput exceeds the given rate.
  int32_t compute_elastic_gpu_target_from_rate(double avg_token_rate);

  // Determine the tag for an elastic pool instance: DECODE if the model has no
  // DECODE instance yet, PREFILL otherwise.
  InstanceTag determine_elastic_tag(const std::string& model_id);

  // Same as determine_elastic_tag but reads instance_tag_map_ directly.
  // Must be called while holding tag_mutex_ (avoids re-entrant locking).
  InstanceTag determine_elastic_tag_locked(const std::string& model_id);

  // Returns true when model has at least one DECODE instance in WAKEUP/ALLOCATED.
  // Must be called while holding tag_mutex_.
  bool has_elastic_decode_slot_locked(const std::string& model_id);

  // Clear stale decode reservation when the model no longer has a DECODE slot.
  // Must be called while holding tag_mutex_.
  void refresh_elastic_decode_slot_locked(const std::string& model_id);

  // Remove decode reservation for a model.
  // Must be called while holding tag_mutex_.
  void clear_elastic_decode_slot_locked(const std::string& model_id,
                                        const std::string& reason);

  // If the model is in the elastic pool and has no DECODE instance, promote
  // one of its PREFILL instances to DECODE.
  void promote_prefill_to_decode(const std::string& model_id);

  // Bidirectional D2D linking between a new instance and its peers (MixPD)
  void link_instance_bidirectional(const std::string& instance_name,
                                   const std::vector<std::string>& peer_names);

 private:

  // send_http_request(instance_name, ...) uses inst_mutex to get_channel()
  // send_http_request(channel, ...) does not use inst_mutex
  bool send_http_request(const std::string& instance_name,
                         const std::string& uri,
                         const std::string& request_body);

  bool send_http_request(std::shared_ptr<brpc::Channel> channel,
                         const std::string& uri,
                         const std::string& request_body,
                         int timeout_ms = -1);

  // Ask a remote janus instance for a locally-free port.
  // Returns the port number, or -1 on failure.
  int request_free_port(std::shared_ptr<brpc::Channel> channel);

  // Periodic low-frequency diagnostics for model P/D allocations.
  void log_model_pd_counts();

  // Periodic diagnostics for total instance counts (model-independent).
  void log_instance_counts();

  // Periodic pool & GPU memory utilization stats (called every 10s).
  void log_pool_memory_stats();

  // Per-instance XTensor heartbeat details for memory leak diagnosis (called every 1s).
  void log_xtensor_heartbeat_details();

 private:
  DISALLOW_COPY_AND_ASSIGN(InstanceMgr);

  // Lock hierarchy (acquire in this order to prevent deadlocks):
  //   1. allocation_mutex_           — outermost, serializes allocation decisions
  //   2. inst_mutex_                 — protects instances_, cached_channels_
  //   3. model_instance_mgr_mutex_  — protects model_instance_mgrs_ map
  //   4. xtensor_info_mutex_         — protects instance_xtensor_infos_
  //   5. pending_mutex_              — protects pending_infos_
  //   6. time_predictor_mutex_       — protects time_predictors_
  //   7. request_metrics_mutex_      — protects request_metrics_
  //   8. latency_metrics_mutex_      — protects latency_metrics_
  //   9. update_mutex_               — protects updated_metrics_, removed_instance_
  //  10. load_metric_mutex_          — protects load_metrics_
  //
  // ModelInstanceMgr internal locks (mutex_, instance_state_all_mutex_,
  // d2d_ref_mutex_, model_heat_mutex_) are leaf-level locks and must not
  // be held when acquiring any InstanceMgr mutex.

  void init();

  bool create_channel(const std::string& target_uri);
  // use etcd as ServiceDiscovery
  void update_instance_metainfo(const etcd::Response& response,
                                const uint64_t& prefix_len);

  void update_load_metrics(const etcd::Response& response,
                           const uint64_t& prefix_len);

  TimePredictor& get_time_predictor(const std::string& instance_name);

  void flip_prefill_to_decode(std::string& instance_name);

  void flip_decode_to_prefill(std::string& instance_name);

  void register_instance(const std::string& instance_name,
                         InstanceMetaInfo metainfo);

 private:
  Options options_;

  bool exited_ = false;
  bool use_etcd_ = false;
  std::atomic_bool is_master_service_ = false;

  std::shared_ptr<EtcdClient> etcd_client_;

  static constexpr double kMaxInstanceMemoryGB = 60.0;
  // model_id -> memory_spec (GB)
  std::unordered_map<std::string, double> model_memory_specs_;

  // instances_ use shared_mutex because they only change when
  // instance registration or model state changes.
  // pending_infos_ use mutex because there's no read-only operations
  
  std::shared_mutex inst_mutex_;
  std::unordered_map<std::string, InstanceMetaInfo> instances_;

  std::shared_mutex model_instance_mgr_mutex_;
  std::unordered_map<std::string, std::shared_ptr<ModelInstanceMgr>> model_instance_mgrs_;

  // stores InstanceMetaInfo before receiving heartbeat
  std::mutex pending_mutex_;
  std::unordered_map<std::string, InstanceMetaInfo> pending_infos_;

  std::shared_mutex load_metric_mutex_;
  std::unordered_map<std::string, LoadMetrics> load_metrics_;
  std::unordered_map<std::string, std::shared_ptr<brpc::Channel>>
      cached_channels_;

  std::mutex update_mutex_;
  std::unordered_map<std::string, LoadMetrics> updated_metrics_;
  std::unordered_set<std::string> removed_instance_;

  // "instance name" -> "TimePredictor" map
  std::mutex time_predictor_mutex_;
  std::unordered_map<std::string, TimePredictor> time_predictors_;

  // Record the latest token latency metrics for each instance, including TTFT
  // and TBT.
  std::mutex latency_metrics_mutex_;
  std::unordered_map<std::string, LatencyMetrics> latency_metrics_;

  // Record the request metrics for each instance, including prefill token
  // count, prefill request count, estimated prefill execution time, decode
  // token count, and decode request count.
  std::mutex request_metrics_mutex_;
  std::unordered_map<std::string, RequestMetrics> request_metrics_;

  // Per-instance list of inflight prefill requests (dispatched but not yet
  // FINISH_PREFILL / CANCEL). Used to propagate EPDT corrections to all
  // remaining requests so that later FINISH_PREFILL does not double-count.
  // Protected by request_metrics_mutex_.
  std::unordered_map<std::string, std::vector<std::shared_ptr<Request>>>
      inflight_prefill_requests_;  
  
  std::mutex allocation_mutex_;

  // Two-pool auto-scaling state
  std::atomic<int32_t> total_available_gpus_{0};
  // model_id -> resource model (read-only after init)
  std::unordered_map<std::string, std::unique_ptr<ResourceModel>> model_resource_models_;
  // model_id -> dynamic pool GP resource model (read-only after init)
  std::unordered_map<std::string, std::unique_ptr<ResourceModel>> dynamic_resource_models_;
  // alias model_id -> real model_id for GP resource model lookup (read-only after init)
  std::unordered_map<std::string, std::string> alias_to_real_model_;
  // GPU hardware spec (read-only after init)
  GpuHardwareSpec gpu_hw_spec_;

  // Resolve alias model_id to real model_id for GP resource model lookup.
  // Returns the real model_id if an alias mapping exists, otherwise returns
  // the input model_id unchanged.
  const std::string& resolve_gp_model_id(const std::string& model_id) const {
    auto it = alias_to_real_model_.find(model_id);
    return (it != alias_to_real_model_.end()) ? it->second : model_id;
  }

  // --- Dual-pool state (protected by allocation_mutex_) ---
  // model_id -> which pool it belongs to
  std::unordered_map<std::string, PoolType> model_pool_assignments_;
  // Steady pool bins (each bin = one instance with co-located models)
  std::vector<SteadyBin> steady_bins_;
  // Instances occupied by elastic pool (one model each)
  std::unordered_set<std::string> elastic_occupied_instances_;
  // Pending steady pool GPU reservations (used during instance reclamation)
  int32_t pending_steady_gpus_ = 0;
  // Repack timer thread
  std::unique_ptr<std::thread> repack_thread_;

  // --- Orphan cleanup state (accessed only from log_model_pd_counts thread) ---
  // Tracks when a model was first detected as orphaned (pool=NONE, heat=0, awake instances)
  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
      orphan_detected_time_;

  // --- Scaling plan anti-jitter state (protected by allocation_mutex_) ---
  struct ScalingPlanEntry {
    int32_t gpu_allocated = 0;
    int64_t heat = 0;
  };
  using ScalingPlan = std::unordered_map<std::string, ScalingPlanEntry>;
  // model_id -> entry from the previous-previous plan
  ScalingPlan last_scaling_plan_;
  // model_id -> entry from the previous plan
  ScalingPlan current_scaling_plan_;
  // Timestamp when current_scaling_plan_ replaced last_scaling_plan_
  std::chrono::steady_clock::time_point last_plan_change_time_;

  bool should_accept_scaling_plan(const ScalingPlan& new_plan) const;

  // --- Elastic-to-steady demotion state (protected by allocation_mutex_) ---
  // model_id -> timestamp when gpu_target first dropped to <= 1
  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
      elastic_low_demand_since_;
  // Demotion check thread
  std::unique_ptr<std::thread> demotion_thread_;
  // Low-frequency model P/D metrics thread
  std::unique_ptr<std::thread> pd_metrics_thread_;

  // Dedicated auto-scaling thread (replaces per-request triggering)
  std::unique_ptr<std::thread> auto_scaling_thread_;
  std::mutex scaling_trigger_mutex_;
  std::condition_variable scaling_trigger_cv_;
  bool scaling_requested_ = false;

  // Condition variable for cold elastic path: signaled when a model wakeup completes
  std::mutex model_wakeup_wait_mutex_;
  std::condition_variable model_wakeup_cv_;

  // Condition variable for steady pool reclaim: signaled when an instance is freed
  std::mutex instance_freed_mutex_;
  std::condition_variable instance_freed_cv_;

  // XTensor memory info per instance
  std::mutex xtensor_info_mutex_;
  std::unordered_map<std::string, InstanceXTensorInfo> instance_xtensor_infos_;

  // Instances that have completed fork_master for all models (ready for D2D linking)
  std::mutex fork_done_mutex_;
  std::unordered_set<std::string> fork_done_instances_;

  // MixPD instance role tags
  mutable std::shared_mutex tag_mutex_;
  std::unordered_map<std::string, InstanceTag> instance_tag_map_;
  // Model-level decode reservation for elastic pool.
  // Prevents concurrent scale-up loops from assigning multiple DECODE tags.
  std::unordered_set<std::string> elastic_decode_reserved_models_;

  ThreadPool threadpool_{64};
};

}  // namespace janus_service
