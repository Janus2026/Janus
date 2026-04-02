#include "instance_mgr.h"

#include <absl/strings/str_join.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <iostream>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <shared_mutex>

#include "scheduler/resource_model/linear_resource_model.h"
#include "scheduler/resource_model/gp_steady_resource_model.h"
#include "scheduler/resource_model/gp_dynamic_resource_model.h"

#include "common/global_gflags.h"
#include "common/types.h"
#include "common/utils.h"
#include "disagg_pd_link.pb.h"

namespace janus_service {

// DEPRECATED: Local port probe is unreliable for remote janus instances
// (TOCTOU race + cross-machine invalidity).  Replaced by
// InstanceMgr::request_free_port() which queries the target machine.
[[maybe_unused]] static int get_free_port() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    LOG(ERROR) << "get_free_port: socket() failed, errno=" << errno;
    return -1;
  }
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(0);
  if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    ::close(fd);
    LOG(ERROR) << "get_free_port: bind() failed, errno=" << errno;
    return -1;
  }
  socklen_t len = sizeof(addr);
  if (getsockname(fd, (struct sockaddr*)&addr, &len) < 0) {
    ::close(fd);
    LOG(ERROR) << "get_free_port: getsockname() failed, errno=" << errno;
    return -1;
  }
  int port = ntohs(addr.sin_port);
  ::close(fd);
  return port;
}

// Compute cosine similarity between item needs vector and bin load vector.
// HBM is normalized to [0,1] by hw capacity; compute_sm and bandwidth are
// already per-GPU fractions [0,1] from the GP model.
// Returns 2.0 (> any valid cosine) if either vector is zero-length.
static double compute_cos_similarity(
    const ResourceNeeds& needs, const SteadyBin& bin,
    const GpuHardwareSpec& hw) {
  // Item vector (all in [0,1])
  double v1 = needs.hbm_gb / hw.hbm_per_gpu_gb;
  double v2 = needs.compute_sm;   // already [0,1] per GPU
  double v3 = needs.bandwidth;    // already [0,1] per GPU

  // Bin load vector: used = capacity - remaining
  double s1 = (hw.hbm_per_gpu_gb - bin.remaining_hbm_gb) / hw.hbm_per_gpu_gb;
  double s2 = 1.0 - bin.remaining_compute_sm;   // remaining is already [0,1]
  double s3 = 1.0 - bin.remaining_bandwidth;     // remaining is already [0,1]

  double dot = v1 * s1 + v2 * s2 + v3 * s3;
  double norm_v = std::sqrt(v1 * v1 + v2 * v2 + v3 * v3);
  double norm_s = std::sqrt(s1 * s1 + s2 * s2 + s3 * s3);

  if (norm_v < 1e-12 || norm_s < 1e-12) return 2.0;
  return dot / (norm_v * norm_s);
}

static std::unordered_map<InstanceType, std::string> ETCD_KEYS_PREFIX_MAP = {
    {InstanceType::DEFAULT, "JANUS:DEFAULT:"},
    {InstanceType::PREFILL, "JANUS:PREFILL:"},
    {InstanceType::DECODE, "JANUS:DECODE:"},
    {InstanceType::MIX, "JANUS:MIX:"},
};
static std::string ETCD_ALL_KEYS_PREFIX = "JANUS:";
static std::string ETCD_LOADMETRICS_PREFIX = "JANUS:LOADMETRICS:";

InstanceMgr::InstanceMgr(const Options& options,
                         const std::shared_ptr<EtcdClient>& etcd_client,
                         const bool is_master_service)
    : options_(options),
      is_master_service_(is_master_service),
      etcd_client_(etcd_client) {
  auto handle_instance_metainfo =
      std::bind(&InstanceMgr::update_instance_metainfo,
                this,
                std::placeholders::_1,
                std::placeholders::_2);
  for (auto& it : ETCD_KEYS_PREFIX_MAP) {
    etcd_client_->add_watch(it.second, handle_instance_metainfo);
  }
  if (!is_master_service_) {
    auto handle_load_metrics = std::bind(&InstanceMgr::update_load_metrics,
                                         this,
                                         std::placeholders::_1,
                                         std::placeholders::_2);
    etcd_client_->add_watch(ETCD_LOADMETRICS_PREFIX, handle_load_metrics);
  }

  init();
}

void InstanceMgr::load_models_config() {
  const std::string& path = FLAGS_models_config_path;
  LOG(INFO) << "Loading models config from: " << path;

  std::ifstream file(path);
  if (!file.is_open()) {
    LOG(FATAL) << "Failed to open models_config_path: " << path;
    return;
  }

  nlohmann::json j;
  try {
    file >> j;
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to parse models_config_path JSON (" << path
               << "): " << e.what();
    return;
  }

  if (!j.is_array()) {
    LOG(FATAL) << "models_config_path JSON must be an array of objects: " << path;
    return;
  }

  MODELS.clear();
  for (const auto& item : j) {
    if (!item.contains("service") || !item.contains("model_path")) {
      LOG(FATAL) << "Each entry in models_config_path must have "
                    "\"service\" and \"model_path\" fields.";
      return;
    }
    MODELS.emplace_back(item["service"].get<std::string>(),
                        item["model_path"].get<std::string>());
  }

  LOG(INFO) << "Loaded " << MODELS.size() << " (service, model_path) pairs "
            << "from " << path;
}

void InstanceMgr::init() {
  // Truncate log files at startup so each run starts fresh.
  for (const auto& path :
       {FLAGS_pool_memory_log_path, FLAGS_check_memory_log_path}) {
    if (!path.empty()) {
      std::ofstream(path, std::ios::trunc);
    }
  }

  load_models_config();
  init_model_memory_specs();
  init_model_resource_coefficients();

  if (!options_.disable_steady_pool()) {
    // Start steady pool repack timer thread
    static constexpr int kRepackIntervalSeconds = 30;
    repack_thread_ = std::make_unique<std::thread>([this]() {
      while (!exited_) {
        std::this_thread::sleep_for(std::chrono::seconds(kRepackIntervalSeconds));
        if (exited_) break;
        steady_part_auto_repacking();
      }
    });

    // Start elastic-to-steady demotion check thread
    static constexpr int kDemotionCheckIntervalSeconds = 10;
    demotion_thread_ = std::make_unique<std::thread>([this]() {
      while (!exited_) {
        std::this_thread::sleep_for(
            std::chrono::seconds(kDemotionCheckIntervalSeconds));
        if (exited_) break;
        elastic_to_steady_demotion();
      }
    });
  }

  // Start dedicated auto-scaling thread (skip if elastic pool disabled)
  if (!options_.disable_elastic_pool()) {
    static constexpr int kAutoScalingIntervalMs = 500;
    auto_scaling_thread_ = std::make_unique<std::thread>([this]() {
      while (!exited_) {
        {
          std::unique_lock<std::mutex> lock(scaling_trigger_mutex_);
          scaling_trigger_cv_.wait_for(
              lock, std::chrono::milliseconds(kAutoScalingIntervalMs),
              [this] { return scaling_requested_ || exited_; });
          scaling_requested_ = false;
        }
        if (exited_) break;
        try_dynamic_part_auto_scaling();
      }
    });
  }

  // Start low-frequency P/D metrics thread.
  static constexpr int kPdMetricsIntervalSeconds = 1;
  static constexpr int kInstanceMetricsIntervalTicks = 60;
  static constexpr int kPoolMemoryStatsIntervalTicks = 1;
  pd_metrics_thread_ = std::make_unique<std::thread>([this]() {
    int tick = 0;
    while (!exited_) {
      std::this_thread::sleep_for(
          std::chrono::seconds(kPdMetricsIntervalSeconds));
      if (exited_) break;
      ++tick;
      if (tick % kPoolMemoryStatsIntervalTicks == 0) {
        log_pool_memory_stats();
      }
      if (tick >= kInstanceMetricsIntervalTicks) {
        log_instance_counts();
        tick = 0;
      }
      log_model_pd_counts();
      log_xtensor_heartbeat_details();
    }
  });

  {
    std::unique_lock<std::shared_mutex> lock(inst_mutex_);
    for (auto& it : ETCD_KEYS_PREFIX_MAP) {
      etcd_client_->get_prefix(it.second, &instances_);
    }
    // create ttft predictor and request metrics for each instance
    {
      std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
      std::lock_guard<std::mutex> request_metrics_lock(request_metrics_mutex_);
      for (auto& pair : instances_) {
        time_predictors_.insert_or_assign(
            pair.first,
            TimePredictor(pair.second.ttft_profiling_data,
                          pair.second.tpot_profiling_data));
        request_metrics_.insert_or_assign(pair.first, RequestMetrics());
      }
    }
    // Sync total_available_gpus_ with instances loaded from etcd, so that
    // subsequent deletes (which call fetch_sub) don't drive the counter negative.
    total_available_gpus_.store(
        static_cast<int32_t>(instances_.size()) * options_.tensor_parallel_size());
    LOG(INFO) << "Load instance info from etcd:" << instances_.size();
    std::vector<std::string> channel_creat_fail_insts;
    for (auto& ist : instances_) {
      if (!create_channel(ist.first)) {
        channel_creat_fail_insts.emplace_back(ist.first);
      } else {
        std::unique_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
        // TODO: support multi-model registration, currently assuming instances serve all models or we need info from somewhere
        // For now, assuming instances serve all known models, or we need to parse from InstanceMetaInfo which model it serves
        // But InstanceMetaInfo struct doesn't have model list. Assuming homogenous cluster or we need to add model info to registration.
        // Assuming simple case: register to all ModelInstanceMgrs corresponding to served models.
        // But wait, InstanceMetaInfo structure is:
        /*
        struct InstanceMetaInfo {
            std::string name;
            std::string ip;
            int32_t port;
            InstanceType type;
            InstanceType current_type;
            int32_t instance_index;
            ...
        };
        */
        // It lacks model info. In multi-model world, we need to know which model an instance serves.
        // The prompt says "string model_id mapped to ModelInstanceMgr".
        // Assuming existing code structure, maybe we register instance to ALL model mgrs?
        // Or we need to look at how `get_next_instance_pair` uses `model_id`.
        // It seems `InstanceMgr` was serving one model type implicitly or mixing them up?
        // Ah, `get_next_instance_pair` takes `model_id`.
        // The original code had `prefill_index_` etc. which were GLOBAL for all models?
        // Yes, "InstanceMgr ... was only supporting one model type".
        // Now "multi-model".
        // We should iterate over supported models and create managers?
        // Or create manager on demand?
        
        // Let's create managers for all hardcoded MODELS for now.
        for (const auto& model_pair : MODELS) {
          std::string model_id = model_pair.first;
          if (model_instance_mgrs_.find(model_id) == model_instance_mgrs_.end()) {
             model_instance_mgrs_[model_id] = std::make_shared<ModelInstanceMgr>(model_id);
          }
          model_instance_mgrs_[model_id]->add_instance(ist.first, ist.second);
        }
      }
    }
    for (auto& name : channel_creat_fail_insts) {
      instances_.erase(name);
      {
        std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
        std::lock_guard<std::mutex> request_metrics_lock(
            request_metrics_mutex_);
        time_predictors_.erase(name);
        request_metrics_.erase(name);
      }
    }
  }
  {
    std::unique_lock<std::shared_mutex> lock(load_metric_mutex_);
    etcd_client_->get_prefix(ETCD_LOADMETRICS_PREFIX, &load_metrics_);
  }

}

InstanceMgr::~InstanceMgr() {
  exited_ = true;
  // Wake the auto-scaling thread so it can exit.
  scaling_trigger_cv_.notify_all();
  if (auto_scaling_thread_ && auto_scaling_thread_->joinable()) {
    auto_scaling_thread_->join();
  }
  if (repack_thread_ && repack_thread_->joinable()) {
    repack_thread_->join();
  }
  if (demotion_thread_ && demotion_thread_->joinable()) {
    demotion_thread_->join();
  }
  if (pd_metrics_thread_ && pd_metrics_thread_->joinable()) {
    pd_metrics_thread_->join();
  }
}

void InstanceMgr::log_instance_counts() {
  int registered = 0;
  int pending = 0;
  {
    std::shared_lock<std::shared_mutex> lock(inst_mutex_);
    registered = static_cast<int>(instances_.size());
  }
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending = static_cast<int>(pending_infos_.size());
  }
  LOG(INFO) << "Instance metrics: registered=" << registered
            << " pending=" << pending
            << " total=" << (registered + pending);
}

void InstanceMgr::log_pool_memory_stats() {
  // 1. Snapshot allocation state (allocation_mutex_ is outermost in hierarchy).
  std::vector<std::string> steady_instance_names;
  std::unordered_set<std::string> elastic_instance_names;
  {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    for (const auto& bin : steady_bins_) {
      steady_instance_names.push_back(bin.instance_name);
    }
    elastic_instance_names = elastic_occupied_instances_;
  }

  // 2. Snapshot XTensor info (xtensor_info_mutex_ is below allocation_mutex_).
  std::unordered_map<std::string, InstanceXTensorInfo> xtensor_snapshot;
  {
    std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
    xtensor_snapshot = instance_xtensor_infos_;
  }

  // 3. Total registered instance count.
  int total_registered = 0;
  {
    std::shared_lock<std::shared_mutex> lock(inst_mutex_);
    total_registered = static_cast<int>(instances_.size());
  }

  const int steady_instances = static_cast<int>(steady_instance_names.size());
  const int steady_gpus = steady_instances * options_.tensor_parallel_size();
  const int elastic_instances = static_cast<int>(elastic_instance_names.size());
  const int elastic_gpus = elastic_instances * options_.tensor_parallel_size();

  // 4. Compute per-pool GPU memory utilization from XTensor heartbeat data.
  const double total_hbm_per_gpu_bytes =
      gpu_hw_spec_.hbm_per_gpu_gb * 1024.0 * 1024.0 * 1024.0;

  uint64_t all_used_bytes = 0;
  int all_gpu_count = 0;

  uint64_t steady_used_bytes = 0;
  int steady_gpu_count = 0;

  uint64_t elastic_used_bytes = 0;
  int elastic_gpu_count = 0;

  const std::unordered_set<std::string> steady_set(
      steady_instance_names.begin(), steady_instance_names.end());

  for (const auto& [inst_name, info] : xtensor_snapshot) {
    if (!info.is_valid) continue;

    const bool is_steady = steady_set.count(inst_name) > 0;
    const bool is_elastic = elastic_instance_names.count(inst_name) > 0;

    for (uint64_t free_pages : info.worker_free_phy_pages) {
      const uint64_t free_bytes = free_pages * kXTensorPageSizeBytes;
      const uint64_t used =
          (total_hbm_per_gpu_bytes > free_bytes)
              ? static_cast<uint64_t>(total_hbm_per_gpu_bytes) - free_bytes
              : 0;
      all_used_bytes += used;
      ++all_gpu_count;

      if (is_steady) {
        steady_used_bytes += used;
        ++steady_gpu_count;
      } else if (is_elastic) {
        elastic_used_bytes += used;
        ++elastic_gpu_count;
      }
    }
  }

  auto to_gb = [](uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  };

  const double total_used_gb = to_gb(all_used_bytes);
  const double steady_avg_gb =
      (steady_gpu_count > 0) ? to_gb(steady_used_bytes) / steady_gpu_count
                             : 0.0;
  const double elastic_avg_gb =
      (elastic_gpu_count > 0) ? to_gb(elastic_used_bytes) / elastic_gpu_count
                              : 0.0;

  const double hbm = gpu_hw_spec_.hbm_per_gpu_gb;
  const double steady_util_pct =
      (hbm > 0.0 && steady_gpu_count > 0) ? steady_avg_gb / hbm * 100.0
                                           : 0.0;
  const double elastic_util_pct =
      (hbm > 0.0 && elastic_gpu_count > 0) ? elastic_avg_gb / hbm * 100.0
                                            : 0.0;
  const double overall_util_pct =
      (hbm > 0.0 && all_gpu_count > 0)
          ? to_gb(all_used_bytes) / all_gpu_count / hbm * 100.0
          : 0.0;

  // 5. Format timestamp.
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) % 1000;
  std::tm tm_buf;
  localtime_r(&time_t_now, &tm_buf);

  std::ostringstream ts;
  ts << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << '.'
     << std::setfill('0') << std::setw(3) << ms.count();

  // 6. Build log lines.
  std::ostringstream line1;
  line1 << "[" << ts.str() << "] "
        << "Pool allocation: total_registered=" << total_registered
        << " steady_instances=" << steady_instances
        << " steady_gpus=" << steady_gpus
        << " elastic_instances=" << elastic_instances
        << " elastic_gpus=" << elastic_gpus;

  std::ostringstream line2;
  line2 << "[" << ts.str() << "] "
        << "Pool memory: total_used_gb=" << std::fixed << std::setprecision(2)
        << total_used_gb
        << " gpus_with_data=" << all_gpu_count
        << " overall_util=" << overall_util_pct << "%"
        << " steady_avg_per_gpu_gb=" << steady_avg_gb
        << " steady_util=" << steady_util_pct << "%"
        << " (gpus=" << steady_gpu_count << ")"
        << " elastic_avg_per_gpu_gb=" << elastic_avg_gb
        << " elastic_util=" << elastic_util_pct << "%"
        << " (gpus=" << elastic_gpu_count << ")"
        << " hbm_per_gpu_gb=" << hbm;

  // 7. Output: dedicated file if configured, otherwise glog.
  if (!FLAGS_pool_memory_log_path.empty()) {
    std::ofstream ofs(FLAGS_pool_memory_log_path, std::ios::app);
    if (ofs.is_open()) {
      ofs << line1.str() << "\n" << line2.str() << "\n";
    } else {
      LOG(WARNING) << "Failed to open pool_memory_log_path: "
                   << FLAGS_pool_memory_log_path;
      LOG(INFO) << line1.str();
      LOG(INFO) << line2.str();
    }
  } else {
    LOG(INFO) << line1.str();
    LOG(INFO) << line2.str();
  }
}

void InstanceMgr::log_xtensor_heartbeat_details() {
  if (FLAGS_check_memory_log_path.empty()) return;

  // 1. Snapshot XTensor info.
  std::unordered_map<std::string, InstanceXTensorInfo> xtensor_snapshot;
  {
    std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
    xtensor_snapshot = instance_xtensor_infos_;
  }

  if (xtensor_snapshot.empty()) return;

  // 2. Snapshot pool assignments.
  std::unordered_set<std::string> steady_names;
  std::unordered_set<std::string> elastic_names;
  {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    for (const auto& bin : steady_bins_) {
      steady_names.insert(bin.instance_name);
    }
    elastic_names = elastic_occupied_instances_;
  }

  // 3. Format timestamp.
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) % 1000;
  std::tm tm_buf;
  localtime_r(&time_t_now, &tm_buf);

  std::ostringstream ts;
  ts << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << '.'
     << std::setfill('0') << std::setw(3) << ms.count();

  const double total_hbm_per_gpu_bytes =
      gpu_hw_spec_.hbm_per_gpu_gb * 1024.0 * 1024.0 * 1024.0;

  auto to_gb = [](uint64_t bytes) -> double {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  };
  auto to_mb = [](uint64_t bytes) -> double {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
  };

  // 4. Build detailed log.
  std::ostringstream oss;
  oss << "========== XTensor Heartbeat Details [" << ts.str()
      << "] instances=" << xtensor_snapshot.size()
      << " hbm_per_gpu=" << std::fixed << std::setprecision(2)
      << gpu_hw_spec_.hbm_per_gpu_gb << "GB ==========\n";

  // Sort instance names for stable output.
  std::vector<std::string> sorted_names;
  sorted_names.reserve(xtensor_snapshot.size());
  for (const auto& [name, _] : xtensor_snapshot) {
    sorted_names.push_back(name);
  }
  std::sort(sorted_names.begin(), sorted_names.end());

  for (const auto& inst_name : sorted_names) {
    const auto& info = xtensor_snapshot.at(inst_name);

    const char* pool = "UNASSIGNED";
    if (steady_names.count(inst_name)) pool = "STEADY";
    else if (elastic_names.count(inst_name)) pool = "ELASTIC";

    oss << "  [Instance] " << inst_name
        << "  pool=" << pool
        << "  valid=" << (info.is_valid ? "true" : "false") << "\n";

    if (!info.is_valid) {
      oss << "    (no valid heartbeat data)\n";
      continue;
    }

    // Per-worker free pages.
    oss << "    Workers(" << info.worker_free_phy_pages.size() << "):";
    for (size_t w = 0; w < info.worker_free_phy_pages.size(); ++w) {
      uint64_t free_pages = info.worker_free_phy_pages[w];
      uint64_t free_bytes = free_pages * kXTensorPageSizeBytes;
      uint64_t used_bytes =
          (total_hbm_per_gpu_bytes > free_bytes)
              ? static_cast<uint64_t>(total_hbm_per_gpu_bytes) - free_bytes
              : 0;
      double util_pct = (total_hbm_per_gpu_bytes > 0)
          ? static_cast<double>(used_bytes) / total_hbm_per_gpu_bytes * 100.0
          : 0.0;
      oss << "  [rank" << w
          << " free_pages=" << free_pages
          << " free=" << std::fixed << std::setprecision(2) << to_gb(free_bytes) << "GB"
          << " used=" << to_gb(used_bytes) << "GB"
          << " util=" << std::setprecision(1) << util_pct << "%]";
    }
    oss << "\n";

    // Aggregate: min free across workers.
    uint64_t min_free = info.get_min_free_bytes();
    oss << "    MinFreeAcrossWorkers: " << std::setprecision(2)
        << to_gb(min_free) << "GB (" << min_free << " bytes)\n";

    // Loaded models and weight segments.
    if (info.model_weight_segments.empty()) {
      oss << "    Models: (none loaded)\n";
    } else {
      oss << "    Models(" << info.model_weight_segments.size() << "):\n";
      for (const auto& [model_id, segments] : info.model_weight_segments) {
        uint64_t total_model_bytes = 0;
        for (const auto& seg : segments) {
          total_model_bytes += seg.size;
        }
        oss << "      model_id=\"" << model_id
            << "\"  segments=" << segments.size()
            << "  total_size=" << to_mb(total_model_bytes) << "MB ("
            << to_gb(total_model_bytes) << "GB)\n";
        for (size_t s = 0; s < segments.size(); ++s) {
          oss << "        seg[" << s
              << "] offset=" << segments[s].offset
              << " size=" << segments[s].size
              << " (" << to_mb(segments[s].size) << "MB)"
              << " end=" << segments[s].end() << "\n";
        }
      }
    }

    // Device & P2P addresses (for D2D context).
    if (!info.device_addrs.empty()) {
      oss << "    DeviceAddrs:";
      for (const auto& addr : info.device_addrs) oss << " " << addr;
      oss << "\n";
    }
    if (!info.p2p_addrs.empty()) {
      oss << "    P2PAddrs:";
      for (const auto& addr : info.p2p_addrs) oss << " " << addr;
      oss << "\n";
    }
  }

  oss << "========== End XTensor Heartbeat Details ==========\n";

  // 5. Write to file.
  std::ofstream ofs(FLAGS_check_memory_log_path, std::ios::app);
  if (ofs.is_open()) {
    ofs << oss.str();
  } else {
    LOG(WARNING) << "Failed to open check_memory_log_path: "
                 << FLAGS_check_memory_log_path;
    LOG(INFO) << oss.str();
  }
}

void InstanceMgr::log_model_pd_counts() {
  // Snapshot model managers to minimize lock hold time.
  std::vector<std::pair<std::string, std::shared_ptr<ModelInstanceMgr>>> mgrs;
  {
    std::shared_lock<std::shared_mutex> lock(model_instance_mgr_mutex_);
    mgrs.reserve(model_instance_mgrs_.size());
    for (const auto& [model_id, mgr] : model_instance_mgrs_) {
      mgrs.emplace_back(model_id, mgr);
    }
  }

  // Snapshot pool assignments for readable logs.
  std::unordered_map<std::string, PoolType> pool_snapshot;
  {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    pool_snapshot = model_pool_assignments_;
  }

  for (const auto& [model_id, mgr] : mgrs) {
    if (!mgr) continue;
    int prefill_count = 0;
    int decode_count = 0;
    int normal_count = 0;
    int64_t heat = mgr->get_model_heat();

    auto all_instances = mgr->get_all_instance_names();
    for (const auto& inst : all_instances) {
      auto state = mgr->get_model_state(inst);
      if (state != ModelState::WAKEUP && state != ModelState::ALLOCATED) {
        continue;
      }
      auto tag = get_instance_tag(inst);
      if (tag == InstanceTag::DECODE) {
        ++decode_count;
      } else if (tag == InstanceTag::PREFILL) {
        ++prefill_count;
      } else if (tag == InstanceTag::NORMAL) {
        ++normal_count;
      }
    }

    const auto it = pool_snapshot.find(model_id);
    const PoolType pool = (it != pool_snapshot.end()) ? it->second : PoolType::NONE;
    const char* pool_name = "NONE";
    if (pool == PoolType::STEADY) {
      pool_name = "STEADY";
    } else if (pool == PoolType::ELASTIC) {
      pool_name = "ELASTIC";
    }

    LOG(INFO) << "Model PD metrics: model=" << model_id
              << " pool=" << pool_name
              << " heat=" << heat
              << " prefill=" << prefill_count
              << " decode=" << decode_count
              << " normal=" << normal_count;
  }

  // --- Orphan cleanup: detect and sleep models stuck in pool=NONE with awake instances ---
  static constexpr int64_t kOrphanGracePeriodSeconds = 30;
  for (const auto& [model_id, mgr] : mgrs) {
    if (!mgr) continue;
    const auto it = pool_snapshot.find(model_id);
    const PoolType pool = (it != pool_snapshot.end()) ? it->second : PoolType::NONE;
    int64_t heat = mgr->get_model_heat();
    auto awake_instances = mgr->get_awake_instances();

    if (pool == PoolType::NONE && heat == 0 && !awake_instances.empty()) {
      auto now = std::chrono::steady_clock::now();
      auto oit = orphan_detected_time_.find(model_id);
      if (oit == orphan_detected_time_.end()) {
        orphan_detected_time_[model_id] = now;
      } else {
        auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(
            now - oit->second).count();
        if (elapsed_s >= kOrphanGracePeriodSeconds) {
          LOG(WARNING) << "Orphan cleanup: model " << model_id
                       << " has pool=NONE, heat=0 but "
                       << awake_instances.size()
                       << " awake instances for >" << elapsed_s
                       << "s, sleeping them";
          for (const auto& inst : awake_instances) {
            send_model_sleep(inst, model_id);
          }
          orphan_detected_time_.erase(model_id);
        }
      }
    } else {
      orphan_detected_time_.erase(model_id);
    }
  }
}

InstanceMetaInfo InstanceMgr::get_instance_info(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Get instance info failed, instance is not registered, "
                  "instance_name: "
               << instance_name;
    return InstanceMetaInfo();
  }
  return instances_[instance_name];
}

bool InstanceMgr::get_next_instance_pair(const std::string& model_id, Routing* routing) {
  std::shared_lock<std::shared_mutex> lock(model_instance_mgr_mutex_);
  auto it = model_instance_mgrs_.find(model_id);
  if (it == model_instance_mgrs_.end()) {
    LOG(ERROR) << "Model manager not found for model " << model_id;
    return false;
  }
  return it->second->get_next_instance_pair(routing);
}

// TODO: refactor later, currently return all decode instances
std::vector<std::string> InstanceMgr::get_static_decode_list(
    const std::string& instance_name) {
  // Logic needs to find which model this instance serves or just return all?
  // Original code returned all DECODE instances.
  // With ModelInstanceMgr, we might need to query specific mgr?
  // But this method takes instance_name... unused?
  // Wait, argument is `instance_name` but not used in search?
  // "currently return all decode instances"
  
  // For now, let's keep it global if possible, OR we need to know the model context.
  // The caller likely wants peers.
  
  std::vector<std::string> decode_list;
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  for (auto& inst : instances_) {
    if (inst.second.type == InstanceType::DECODE ||
        inst.second.type == InstanceType::MIX) {
      decode_list.emplace_back(inst.second.name);
    }
  }
  return decode_list;
}

// TODO: refactor later, currently return all prefill instances
std::vector<std::string> InstanceMgr::get_static_prefill_list(
    const std::string& instance_name) {
  std::vector<std::string> prefill_list;
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  for (auto& inst : instances_) {
    if (inst.second.type == InstanceType::PREFILL ||
        inst.second.type == InstanceType::DEFAULT ||
        inst.second.type == InstanceType::MIX) {
      prefill_list.emplace_back(inst.second.name);
    }
  }
  return prefill_list;
}

void InstanceMgr::fork_master_and_sleep(
    const std::string& instance_name,
    std::shared_ptr<brpc::Channel> channel) {
  LOG(INFO) << "Forking master and sleeping for instance " << instance_name;

  if (!FLAGS_initial_model_id.empty()) {
    LOG(INFO) << "Sleeping initial model " << FLAGS_initial_model_id
              << " on instance " << instance_name
              << " to free GPU memory before fork";
    nlohmann::json sleep_body;
    sleep_body["model_id"] = FLAGS_initial_model_id;
    sleep_body["master_status"] = 2;  // LIGHT_SLEEP
    static constexpr int kMaxSleepRetries = 10;
    bool sleep_succeeded = false;
    for (int attempt = 0; attempt < kMaxSleepRetries && !sleep_succeeded;
         ++attempt) {
      if (send_http_request(channel, "/sleep", sleep_body.dump())) {
        sleep_succeeded = true;
      } else {
        LOG(WARNING) << "Failed to sleep initial model " << FLAGS_initial_model_id
                     << " on instance " << instance_name
                     << ", attempt " << attempt + 1 << "/" << kMaxSleepRetries;
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }
    if (!sleep_succeeded) {
      LOG(ERROR) << "Failed to sleep initial model " << FLAGS_initial_model_id
                 << " on instance " << instance_name
                 << " after " << kMaxSleepRetries << " retries";
    }
  }

  for (const auto& model : MODELS) {
    // 1. Fork Master

    auto model_id = model.first;

    /* hardcoded for now */
    int base_port = stoi(instance_name.substr(instance_name.find(":") + 1));

    LOG(INFO) << "Forking master and sleeping for model " << model_id << " on instance " << instance_name;

    static constexpr int kMaxForkRetries = 10;
    bool fork_succeeded = false;

    for (int attempt = 0; attempt < kMaxForkRetries && !fork_succeeded; ++attempt) {
      // Allocate port on node 0's machine (the one that binds CollectiveServer).
      // For TP=1 this is the only node; for TP>1 it is the base instance.
      int master_port = request_free_port(channel);
      if (master_port < 0) {
        LOG(ERROR) << "Failed to get free port from remote " << instance_name
                   << " for model " << model_id
                   << ", attempt " << attempt + 1 << "/" << kMaxForkRetries;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        continue;
      }

      nlohmann::json fork_body;
      fork_body["model_id"] = model_id;
      fork_body["model_path"] = model.second;
      fork_body["master_node_addr"] = "127.0.0.1:" + std::to_string(master_port);
      fork_body["master_status"] = 1;
      fork_body["nnodes"] = options_.tensor_parallel_size();

      std::vector<std::thread> fork_threads;
      std::atomic<int> fork_success_count(0);

      for (int node_idx = 0; node_idx < options_.tensor_parallel_size();
           ++node_idx) {

        /* hardcoded for now */
        int tmp_port = base_port + node_idx;
        std::string tmp_instance_name = instance_name.substr(0, instance_name.find(":")) +
                                        ":" + std::to_string(tmp_port);
        std::shared_ptr<brpc::Channel> tmp_channel = get_channel(tmp_instance_name);
        if (!tmp_channel) {
          LOG(ERROR) << "Channel not found for " << tmp_instance_name
                     << ", skipping fork for model " << model_id;
          continue;
        }

        fork_threads.emplace_back([this, tmp_instance_name, node_idx, fork_body, model_id, tmp_channel, &fork_success_count]() {
          if (node_idx > 0) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
          }
          static constexpr int kForkTimeoutMs = 600000;
          if (send_http_request(tmp_channel, "/fork_master",
                                fork_body.dump(), kForkTimeoutMs)) {
            fork_success_count += 1;
          } else {
            LOG(WARNING) << "Failed to fork master for model " << model_id
                         << " on " << tmp_instance_name;
          }
        });
      }

      for (auto& t : fork_threads) {
        if (t.joinable()) {
          t.join();
        }
      }

      if (fork_success_count.load() == options_.tensor_parallel_size()) {
        fork_succeeded = true;
        LOG(INFO) << "Fork master succeeded for model " << model_id
                  << " on instance " << instance_name
                  << " (master_port=" << master_port << ")";
      } else {
        LOG(WARNING) << "Fork master failed for model " << model_id
                     << " on " << instance_name
                     << " (master_port=" << master_port << "), retry " << attempt + 1
                     << "/" << kMaxForkRetries
                     << " (will request new port from remote)";
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }

    if (!fork_succeeded) {
      LOG(ERROR) << "Failed to fork master for model " << model.first << " on "
                << instance_name << " after " << kMaxForkRetries << " retries";
      continue;
    }

    auto mgr = get_model_instance_mgr(model.first);
    mgr->set_model_state(instance_name, ModelState::SLEEP);

    // 2. force sleep (the initial model of janus instance fails to fork_master)
    nlohmann::json sleep_body;
    sleep_body["model_id"] = model.first;
    sleep_body["master_status"] = 1;
    
    send_http_request(channel, "/sleep", sleep_body.dump());
  }

  LOG(INFO) << "All models fork_master'd — bidirectional D2D linking with all ready instances";

  // All models fork_master'd — bidirectional D2D linking with all ready instances
  {
    auto new_info = get_instance_xtensor_info(instance_name);
    LOG(INFO) << "New instance " << instance_name << " has " << new_info->device_addrs.size() << " device addrs";
    if (new_info && !new_info->device_addrs.empty()) {
      // Snapshot fork_done instances (release lock before acquiring xtensor_info_mutex_)
      std::vector<std::string> done_peers;
      {
        std::lock_guard<std::mutex> lock(fork_done_mutex_);
        done_peers.assign(fork_done_instances_.begin(),
                          fork_done_instances_.end());
      }

      // Gather peer channels and device addrs
      std::vector<std::pair<std::shared_ptr<brpc::Channel>,
                             std::vector<std::string>>> peers;
      {
        std::lock_guard<std::mutex> xt_lock(xtensor_info_mutex_);
        for (const auto& peer_name : done_peers) {
          auto it = instance_xtensor_infos_.find(peer_name);
          if (it != instance_xtensor_infos_.end() && it->second.is_valid &&
              !it->second.device_addrs.empty()) {
            auto peer_channel = get_channel(peer_name);
            if (peer_channel) {
              peers.emplace_back(peer_channel, it->second.device_addrs);
            }
          }
        }
      }

      // Use first model's mgr for the link call (mooncake session is model-agnostic)
      auto mgr = get_model_instance_mgr(MODELS[0].first);
      LOG(INFO) << "Linking D2D bidirectional for model " << MODELS[0].first << " with " << peers.size() << " peers";
      if (mgr) {
        mgr->link_d2d_bidirectional(channel, new_info->device_addrs, peers);
      }

      // Mark self as fork_done
      {
        std::lock_guard<std::mutex> lock(fork_done_mutex_);
        fork_done_instances_.insert(instance_name);
      }
      LOG(INFO) << "Instance " << instance_name
                << " completed all model forks and D2D linking";
    } else {
      LOG(WARNING) << "No device addrs for instance " << instance_name
                   << ", skipping D2D linking";
      // Still mark as fork_done so others can link to us later
      std::lock_guard<std::mutex> lock(fork_done_mutex_);
      fork_done_instances_.insert(instance_name);
    }
  }

  // Bidirectional DisaggPD RPC linking with all ready peers
  {
    std::vector<std::string> done_peers;
    {
      std::lock_guard<std::mutex> lock(fork_done_mutex_);
      done_peers.assign(fork_done_instances_.begin(),
                        fork_done_instances_.end());
    }
    link_instance_bidirectional(instance_name, done_peers);
  }
}

bool InstanceMgr::send_http_request(const std::string& instance_name,
                                    const std::string& uri,
                                    const std::string& request_body) {
  std::shared_ptr<brpc::Channel> channel = get_channel(instance_name);
  if (!channel) {
    LOG(ERROR) << "Channel not found for " << instance_name;
    return false;
  }
  return send_http_request(channel, uri, request_body);
}

bool InstanceMgr::send_http_request(std::shared_ptr<brpc::Channel> channel,
                                    const std::string& uri,
                                    const std::string& request_body,
                                    int timeout_ms) {
  brpc::Controller cntl;
  cntl.http_request().uri() = uri;  // brpc channel already has host:port
  cntl.http_request().set_method(brpc::HTTP_METHOD_POST);
  cntl.http_request().set_content_type("application/json");
  cntl.request_attachment().append(request_body);
  if (timeout_ms > 0) {
    cntl.set_timeout_ms(timeout_ms);
  }

  channel->CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);

  if (cntl.Failed()) {
    LOG(ERROR) << "HTTP request failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

int InstanceMgr::request_free_port(std::shared_ptr<brpc::Channel> channel) {
  brpc::Controller cntl;
  cntl.http_request().uri() = "/get_free_port";
  cntl.http_request().set_method(brpc::HTTP_METHOD_GET);
  cntl.set_timeout_ms(5000);

  channel->CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);

  if (cntl.Failed()) {
    LOG(ERROR) << "request_free_port RPC failed: " << cntl.ErrorText();
    return -1;
  }

  try {
    auto body = cntl.response_attachment().to_string();
    auto resp = nlohmann::json::parse(body);
    int port = resp.value("port", -1);
    if (port <= 0) {
      LOG(ERROR) << "Remote returned invalid port: " << body;
    }
    return port;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to parse get_free_port response: " << e.what();
    return -1;
  }
}

void InstanceMgr::get_load_metrics(LoadBalanceInfos* infos) {
  std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
  std::shared_lock<std::shared_mutex> metric_lock(load_metric_mutex_);

  for (auto name : infos->overlap_scores.instances) {
    auto it = load_metrics_.find(name);
    if (it == load_metrics_.end()) {
      continue;
    }
    auto instance_it = instances_.find(name);
    if (instance_it == instances_.end()) {
      continue;
    }

    if (instance_it->second.type == InstanceType::DECODE) {
      infos->decode_load_metrics.insert(std::make_pair(name, it->second));
      infos->decode_max_waiting_requests_num =
          std::max(infos->decode_max_waiting_requests_num,
                   it->second.waiting_requests_num);
    } else {
      infos->prefill_load_metrics.insert(std::make_pair(name, it->second));
      infos->prefill_max_waiting_requests_num =
          std::max(infos->prefill_max_waiting_requests_num,
                   it->second.waiting_requests_num);
    }
  }

  std::string least_loaded_prefill_instance;
  float least_loaded_prefill_gpu_cache_usage_perc = 1;
  std::string least_loaded_decode_instance;
  float least_loaded_decode_gpu_cache_usage_perc = 1;

  if (infos->prefill_load_metrics.size() == 0 ||
      infos->decode_load_metrics.size() == 0) {
    for (const auto& metric : load_metrics_) {
      auto instance_it = instances_.find(metric.first);
      if (instance_it != instances_.end()) {
        if (instance_it->second.type != InstanceType::DECODE) {
          if (metric.second.gpu_cache_usage_perc <
              least_loaded_prefill_gpu_cache_usage_perc) {
            least_loaded_prefill_gpu_cache_usage_perc =
                metric.second.gpu_cache_usage_perc;
            least_loaded_prefill_instance = metric.first;
          }
        } else {
          if (metric.second.gpu_cache_usage_perc <
              least_loaded_decode_gpu_cache_usage_perc) {
            least_loaded_decode_gpu_cache_usage_perc =
                metric.second.gpu_cache_usage_perc;
            least_loaded_decode_instance = metric.first;
          }
        }
      }
    }
  }

  if (infos->prefill_load_metrics.size() == 0 &&
      !least_loaded_prefill_instance.empty()) {
    infos->prefill_load_metrics.insert(
        std::make_pair(least_loaded_prefill_instance,
                       load_metrics_[least_loaded_prefill_instance]));
  }

  if (infos->decode_load_metrics.size() == 0 &&
      !least_loaded_decode_instance.empty()) {
    infos->decode_load_metrics.insert(
        std::make_pair(least_loaded_decode_instance,
                       load_metrics_[least_loaded_decode_instance]));
  }
}

void InstanceMgr::record_load_metrics_update(
    const std::string& instance_name,
    const proto::LoadMetrics& load_metrics) {
  std::lock_guard<std::mutex> lock(update_mutex_);

  updated_metrics_.insert_or_assign(
      instance_name,
      LoadMetrics(load_metrics.waiting_requests_num(),
                  load_metrics.gpu_cache_usage_perc()));
}

bool InstanceMgr::upload_load_metrics() {
  std::lock_guard<std::mutex> lock(update_mutex_);
  bool status = etcd_client_->set(ETCD_LOADMETRICS_PREFIX, updated_metrics_);
  status =
      status && etcd_client_->rm(ETCD_LOADMETRICS_PREFIX, removed_instance_);
  {
    std::unique_lock<std::shared_mutex> lock(inst_mutex_);
    for (auto& iter : updated_metrics_) {
      load_metrics_.insert_or_assign(iter.first, std::move(iter.second));
    }
    for (auto& iter : removed_instance_) {
      load_metrics_.erase(iter);
    }
  }
  updated_metrics_.clear();
  removed_instance_.clear();

  return status;
}

void InstanceMgr::set_as_master() {
  is_master_service_ = true;
  etcd_client_->remove_watch(ETCD_LOADMETRICS_PREFIX);
}

void InstanceMgr::on_heartbeat(const std::string& instance_name) {
  InstanceMetaInfo metainfo;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_infos_.find(instance_name);
    if (it == pending_infos_.end()) {
      return;
    }
    metainfo = std::move(it->second);
    pending_infos_.erase(it);
  }

  LOG(INFO) << "Received heartbeat from pending instance: " << instance_name;
  threadpool_.schedule([this, instance_name, metainfo = std::move(metainfo)]() {
    register_instance(instance_name, metainfo);
  });
}

void InstanceMgr::register_instance(const std::string& instance_name,
                                    InstanceMetaInfo metainfo) {
  std::unique_lock<std::shared_mutex> lock(inst_mutex_);
  if (instances_.find(instance_name) != instances_.end()) {
    LOG(ERROR) << "Instance is already registered, instance_name: "
               << instance_name;
    return;
  }
  
  if (!create_channel(instance_name)) {
    LOG(ERROR) << "create channel fail: " << instance_name;
    return;
  }

  /* hardcoded for now*/
  if (options_.tensor_parallel_size() > 1) {
    for (int node_idx = 1; node_idx < options_.tensor_parallel_size();
         ++node_idx) {
      int instance_port = stoi(instance_name.substr(instance_name.find(":") + 1));
      int tmp_port = instance_port + node_idx;
      std::string tmp_instance_name = instance_name.substr(0, instance_name.find(":")) +
                                      ":" + std::to_string(tmp_port);
      if (!create_channel(tmp_instance_name)) {
        LOG(ERROR) << "create channel fail: " << tmp_instance_name;
        return;
      }
    }
  }
  
  // Note: we can't call fork_master_and_sleep here if we are holding
  // inst_mutex_ and fork_master_and_sleep calls send_http_request which calls
  // get_channel which acquires inst_mutex_ again (deadlock).
  auto channel = cached_channels_[instance_name];
  threadpool_.schedule([this, instance_name, channel]() {
    fork_master_and_sleep(instance_name, channel);
  });

  // Initialize xtensor info as invalid only if not already set
  // (update_xtensor_info from the same heartbeat may have already set a valid entry)
  {
    std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
    if (instance_xtensor_infos_.find(instance_name) == instance_xtensor_infos_.end()) {
      InstanceXTensorInfo info;
      info.is_valid = false;  // Mark as not yet received valid data
      // Populate device_addrs from registration info for early D2D linking
      info.device_addrs = metainfo.device_addrs;
      info.p2p_addrs = metainfo.p2p_addrs;
      instance_xtensor_infos_[instance_name] = std::move(info);
    } else if (instance_xtensor_infos_[instance_name].device_addrs.empty()) {
      // Entry exists but missing device_addrs — fill from registration
      instance_xtensor_infos_[instance_name].device_addrs = metainfo.device_addrs;
      instance_xtensor_infos_[instance_name].p2p_addrs = metainfo.p2p_addrs;
    }
  }

  {
    std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
    std::lock_guard<std::mutex> request_metrics_lock(request_metrics_mutex_);
    // create ttft predictor for instance
    time_predictors_.emplace(instance_name,
                             TimePredictor(metainfo.ttft_profiling_data,
                                           metainfo.tpot_profiling_data));

    // create request metrics for instance
    request_metrics_.emplace(instance_name, RequestMetrics());
    for (auto& model : MODELS) {
      request_metrics_[instance_name].model_metrics.try_emplace(model.first);
    }
  }

  // Register with ModelInstanceMgrs
  {
    std::unique_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
    for (const auto& model_pair : MODELS) {
      std::string model_id = model_pair.first;
      if (model_instance_mgrs_.find(model_id) == model_instance_mgrs_.end()) {
          model_instance_mgrs_[model_id] = std::make_shared<ModelInstanceMgr>(model_id);
      }
      model_instance_mgrs_[model_id]->add_instance(instance_name, metainfo);
    }
  }

  // Legacy code cleanup / adaptation
  // Note: original code updated indices here. ModelInstanceMgr does it internally.
  // We still need to respect InstanceType logging if needed.
  
  /*
  switch (metainfo.type) {
    // ...
  }
  */
  // Since we moved logic to ModelInstanceMgr, we might just log here.
  LOG(INFO) << "Registered instance " << instance_name << " type " << (int)metainfo.type;

  instances_.insert(std::make_pair(instance_name, std::move(metainfo)));
  total_available_gpus_.fetch_add(options_.tensor_parallel_size());
}

std::shared_ptr<brpc::Channel> InstanceMgr::get_channel(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  auto iter = cached_channels_.find(instance_name);
  if (iter == cached_channels_.end()) {
    return nullptr;
  }
  return iter->second;
}

bool InstanceMgr::create_channel(const std::string& instance_name) {
  if (cached_channels_.find(instance_name) == cached_channels_.end()) {
    auto channel = std::make_shared<brpc::Channel>();
    brpc::ChannelOptions options;
    // Add to params
    options.protocol = "http";
    options.timeout_ms = options_.timeout_ms(); /*milliseconds*/
    options.max_retry = 3;
    std::string load_balancer = "";
    if (channel->Init(instance_name.c_str(), load_balancer.c_str(), &options) !=
        0) {
      LOG(ERROR) << "Fail to initialize channel for " << instance_name;
      return false;
    }
    cached_channels_[instance_name] = std::move(channel);
  }

  return true;
}

void InstanceMgr::update_instance_metainfo(const etcd::Response& response,
                                           const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }

  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    std::unordered_map<std::string, InstanceMetaInfo> put_map;
    std::vector<std::string> delete_list;

    for (const auto& event : response.events()) {
      std::string instance_name = event.kv().key().substr(prefix_len);

      if (event.event_type() == etcd::Event::EventType::PUT) {
        InstanceMetaInfo metainfo;
        auto json_str = event.kv().as_string();
        if (!metainfo.parse_from_json(json_str)) {
          LOG(ERROR) << "pase json:" << json_str << " error!";
          continue;
        }
        put_map.insert(std::make_pair(instance_name, std::move(metainfo)));

      } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
        delete_list.push_back(instance_name);
      }
    }

    {
      std::unique_lock<std::shared_mutex> lock(inst_mutex_);
      for (auto& iter : put_map) {
        if (instances_.find(iter.first) != instances_.end()) {
          // Update existing instance profiling data
          LOG(INFO) << "Update instance profiling data, instance_name: " << iter.first;
          auto& exist_info = instances_[iter.first];
          auto& new_info = iter.second;
          
          // Merge per-model DisaggPD RPC addresses
          for (auto& [mid, addr] : new_info.disagg_pd_rpc_addresses) {
            exist_info.disagg_pd_rpc_addresses[mid] = addr;
          }

          // Merge TTFT profiling data
          for (auto& [model_id, data] : new_info.ttft_profiling_data) {
            exist_info.ttft_profiling_data[model_id] = std::move(data);
          }
          
          // Merge TPOT profiling data
          for (auto& [model_id, data] : new_info.tpot_profiling_data) {
            exist_info.tpot_profiling_data[model_id] = std::move(data);
          }

          // Update TimePredictor
          {
            std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
            time_predictors_.insert_or_assign(
                iter.first,
                TimePredictor(exist_info.ttft_profiling_data,
                              exist_info.tpot_profiling_data));
          }
          continue;
        }

        {
          std::lock_guard<std::mutex> lock(pending_mutex_);
          if (pending_infos_.count(iter.first)) {
            auto& pending = pending_infos_[iter.first];
            for (auto& [mid, addr] : iter.second.disagg_pd_rpc_addresses) {
              pending.disagg_pd_rpc_addresses[mid] = addr;
            }
            LOG(INFO) << "Merged disagg_pd_rpc_addresses for pending instance: "
                      << iter.first;
            continue;
          }
          pending_infos_.insert(
              std::make_pair(iter.first, std::move(iter.second)));
          LOG(INFO) << "Add instance to pending list and wait for heartbeat: "
                    << iter.first;
        }
      }

      for (auto& iter : delete_list) {
        LOG(INFO) << "delete instance: " << iter;
        {
          std::lock_guard<std::mutex> lock(pending_mutex_);
          if (pending_infos_.count(iter)) {
            pending_infos_.erase(iter);
            LOG(INFO) << "Delete pending instance: " << iter;
            continue;
          }
        }
        if (instances_.find(iter) == instances_.end()) {
          LOG(ERROR) << "Instance is already deleted, instance_name: " << iter;
          continue;
        }
        // TODO: notify cache manager to clear expire cache
        // Remove from ModelInstanceMgrs
        {
          std::unique_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
          for (auto& pair : model_instance_mgrs_) {
            pair.second->remove_instance(iter);
          }
        }

        instances_.erase(iter);
        cached_channels_.erase(iter);
        total_available_gpus_.fetch_sub(options_.tensor_parallel_size());
        {
          std::lock_guard<std::mutex> time_predictor_lock(
              time_predictor_mutex_);
          std::lock_guard<std::mutex> request_metrics_lock(
              request_metrics_mutex_);
          time_predictors_.erase(iter);
          request_metrics_.erase(iter);
        }
        {
          std::lock_guard<std::mutex> lock(update_mutex_);
          updated_metrics_.erase(iter);
          removed_instance_.insert(iter);
        }
      }
    }
  });
}

void InstanceMgr::update_load_metrics(const etcd::Response& response,
                                      const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }
  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    std::unordered_map<std::string, LoadMetrics> put_map;
    std::vector<std::string> delete_list;

    for (const auto& event : response.events()) {
      std::string instance_name = event.kv().key().substr(prefix_len);

      if (event.event_type() == etcd::Event::EventType::PUT) {
        LoadMetrics load_metrics;
        auto json_str = event.kv().as_string();
        if (!load_metrics.parse_from_json(json_str)) {
          LOG(ERROR) << "pase json:" << json_str << " error!";
          continue;
        }

        put_map.insert(std::make_pair(instance_name, std::move(load_metrics)));

      } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
        delete_list.push_back(instance_name);
      }
    }

    {
      std::unique_lock<std::shared_mutex> lock(load_metric_mutex_);
      for (auto& iter : put_map) {
        load_metrics_.insert_or_assign(iter.first, std::move(iter.second));
      }

      for (auto& iter : delete_list) {
        load_metrics_.erase(iter);
      }
    }
  });
}

void InstanceMgr::update_latency_metrics(
    const std::string& instance_name,
    const proto::LatencyMetrics& latency_metrics) {
  std::lock_guard<std::mutex> lock(latency_metrics_mutex_);

  LatencyMetrics metrics;
  for (const auto& entry : latency_metrics.model_metrics()) {
    const std::string& model_id = entry.first;
    const auto& proto_model_metrics = entry.second;

    ModelLatencyMetrics model_metrics;
    model_metrics.recent_max_ttft = proto_model_metrics.recent_max_ttft();
    model_metrics.recent_max_tbt = proto_model_metrics.recent_max_tbt();
    
    metrics.model_metrics[model_id] = model_metrics;
  }
  
  latency_metrics_.insert_or_assign(instance_name, std::move(metrics));
}

void InstanceMgr::update_request_metrics(std::shared_ptr<Request> request,
                                         RequestAction action) {
  std::lock_guard<std::mutex> lock(request_metrics_mutex_);

  auto prefill_instance_it = request_metrics_.find(request->routing.prefill_name);
  if (prefill_instance_it == request_metrics_.end()) {
    LOG(ERROR) << "Failed to find prefill instance request metrics, instance name : "
               << request->routing.prefill_name;
    return;
  }

  auto prefill_model_it = prefill_instance_it->second.model_metrics.find(
      request->model);
  if (prefill_model_it == prefill_instance_it->second.model_metrics.end()) {
    LOG(ERROR) << "Failed to find prefill model request metrics, instance name : "
               << request->routing.prefill_name
               << ", model id : " << request->model;
    return;
  }

  if (request->routing.decode_name.empty()) {
    request->routing.decode_name = request->routing.prefill_name;
  }

  auto decode_instance_it = request_metrics_.find(request->routing.decode_name);
  if (decode_instance_it == request_metrics_.end()) {
    LOG(ERROR) << "Failed to find decode instance request metrics, instance name : "
               << request->routing.decode_name;
    return;
  }

  auto decode_model_it = decode_instance_it->second.model_metrics.find(
      request->model);
  if (decode_model_it == decode_instance_it->second.model_metrics.end()) {
    LOG(ERROR) << "Failed to find decode model request metrics, instance name : "
               << request->routing.decode_name
               << ", model id : " << request->model;
    return;
  }

  int64_t num_prompt_tokens = request->token_ids.size();
  int64_t num_generated_tokens = request->num_generated_tokens;
  switch (action) {
    case RequestAction::SCHEDULE:
      // update the request metrics for prefill and decode instances when
      // request is scheduled
      prefill_model_it->second.prefill_request_num += 1;
      prefill_model_it->second.prefill_token_num += num_prompt_tokens;

      decode_model_it->second.decode_request_num += 1;
      decode_model_it->second.decode_token_num += num_prompt_tokens;

      // Update estimated_prefill_done_time if not already set by SLO_AWARE selection
      if (request->expected_prefill_done_ms == 0) {
        int64_t now_ms = absl::ToUnixMillis(absl::Now());
        auto& epdt = prefill_instance_it->second.estimated_prefill_done_time;
        epdt = std::max(epdt, now_ms) + request->estimated_ttft;
        request->expected_prefill_done_ms = epdt;
      }

      // Track this request for EPDT correction propagation
      inflight_prefill_requests_[request->routing.prefill_name].push_back(request);
      break;
    case RequestAction::FINISH_PREFILL:
      // update the request metrics for prefill and decode instance when request
      // finishes the prefill phase
      prefill_model_it->second.prefill_request_num -= 1;
      prefill_model_it->second.prefill_token_num -= num_prompt_tokens;

      // Apply correction feedback to estimated_prefill_done_time
      {
        int64_t now_ms = absl::ToUnixMillis(absl::Now());
        auto& epdt = prefill_instance_it->second.estimated_prefill_done_time;
        if (request->expected_prefill_done_ms > 0) {
          // Shift EPDT by the difference between actual and expected completion
          int64_t correction = now_ms - request->expected_prefill_done_ms;
          epdt += correction;

          // Propagate correction to all remaining inflight requests on this
          // instance so their future FINISH_PREFILL won't double-count.
          auto& inflight =
              inflight_prefill_requests_[request->routing.prefill_name];
          for (auto& req : inflight) {
            if (req->service_request_id != request->service_request_id) {
              req->expected_prefill_done_ms += correction;
            }
          }
        }
        // Clamp: EPDT should never be in the past
        epdt = std::max(epdt, now_ms);

        // Remove this request from inflight list
        auto& inflight =
            inflight_prefill_requests_[request->routing.prefill_name];
        inflight.erase(
            std::remove_if(
                inflight.begin(), inflight.end(),
                [&](const std::shared_ptr<Request>& r) {
                  return r->service_request_id ==
                         request->service_request_id;
                }),
            inflight.end());
      }

      decode_model_it->second.decode_token_num += 1;
      break;
    case RequestAction::GENERATE:
      // update the request metrics for decode instance when request generate a
      // token
      decode_model_it->second.decode_token_num += 1;
      break;
    case RequestAction::FINISH_DECODE:
      // update the request metrics for decode instance when request finishes
      // the decode phase
      decode_model_it->second.decode_request_num -= 1;
      decode_model_it->second.decode_token_num -=
          (num_prompt_tokens + num_generated_tokens);
      break;
    case RequestAction::CANCEL:
      // update the request metrics for prefill and decode instances when
      // request is cancelled
      prefill_model_it->second.prefill_request_num -= 1;
      prefill_model_it->second.prefill_token_num -= num_prompt_tokens;

      // Remove this request's contribution from EPDT
      {
        auto& epdt = prefill_instance_it->second.estimated_prefill_done_time;
        int64_t removed_ttft = request->estimated_ttft;
        epdt -= removed_ttft;
        int64_t now_ms = absl::ToUnixMillis(absl::Now());
        // Clamp: EPDT should never be in the past
        epdt = std::max(epdt, now_ms);

        // Propagate removal to remaining inflight requests: their expected
        // times shift earlier by the cancelled request's contribution.
        auto& inflight =
            inflight_prefill_requests_[request->routing.prefill_name];
        for (auto& req : inflight) {
          if (req->service_request_id != request->service_request_id) {
            req->expected_prefill_done_ms -= removed_ttft;
          }
        }

        // Remove this request from inflight list
        inflight.erase(
            std::remove_if(
                inflight.begin(), inflight.end(),
                [&](const std::shared_ptr<Request>& r) {
                  return r->service_request_id ==
                         request->service_request_id;
                }),
            inflight.end());
      }

      decode_model_it->second.decode_request_num -= 1;
      decode_model_it->second.decode_token_num -=
          (num_prompt_tokens + num_generated_tokens);
      break;
    default:
      LOG(ERROR) << "Unknown RequestAction: " << static_cast<int32_t>(action);
      break;
  }

  if (action == RequestAction::FINISH_PREFILL ||
      action == RequestAction::FINISH_DECODE ||
      action == RequestAction::CANCEL) {

    if (prefill_model_it->second.prefill_request_num == 0 &&
        prefill_model_it->second.decode_request_num == 0) {
      prefill_model_it->second.cv_idle.notify_all();
    }

    if (request->routing.prefill_name != request->routing.decode_name &&
        decode_model_it->second.prefill_request_num == 0 &&
        decode_model_it->second.decode_request_num == 0) {
      decode_model_it->second.cv_idle.notify_all();
    }

  }

  /*
  if (options_.load_balance_policy() == "SLO_AWARE" &&
      decode_it->second.decode_request_num == 0) {
    std::unique_lock<std::shared_mutex> instance_lock(inst_mutex_);
    flip_decode_to_prefill(request->routing.decode_name);
  }
  */
}

bool InstanceMgr::select_instance_pair_on_slo(
    std::shared_ptr<Request> request) {
  std::unique_lock<std::shared_mutex> lock(inst_mutex_);
  std::lock_guard<std::mutex> request_metrics_lock(request_metrics_mutex_);
  auto awake_instances = get_awake_instances(request->model);
  if (awake_instances.empty()) {
    LOG(ERROR) << "No awake instance found for model " << request->model;
    return false;
  }

  // get earliest estimated_prefill_done_time instance from request metrics
  auto best_instance = awake_instances[0];
  int64_t min_done_time = std::numeric_limits<int64_t>::max();
  for (auto& instance : awake_instances) {
    int64_t done_time = request_metrics_[instance].estimated_prefill_done_time;
    if (done_time < min_done_time) {
      best_instance = instance;
      min_done_time = done_time;
    }
  }

  request->routing.prefill_name = best_instance;
  request->routing.decode_name = best_instance;
  request->estimated_ttft =
      predict_ttft(best_instance, request->model, request->token_ids.size());

  // Update EPDT immediately to prevent concurrent selections picking same instance
  {
    int64_t now_ms = absl::ToUnixMillis(absl::Now());
    auto& epdt = request_metrics_[best_instance].estimated_prefill_done_time;
    epdt = std::max(epdt, now_ms) + request->estimated_ttft;
    request->expected_prefill_done_ms = epdt;
  }

  return true;
}

int64_t InstanceMgr::get_estimated_prefill_done_time(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(request_metrics_mutex_);
  auto it = request_metrics_.find(instance_name);
  if (it != request_metrics_.end()) {
    return it->second.estimated_prefill_done_time;
  }
  return 0;
}

// flip all models
void InstanceMgr::flip_prefill_to_decode(std::string& instance_name) {
  {
    std::shared_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
    // Flip in all managers
    for (auto& pair : model_instance_mgrs_) {
      pair.second->flip_prefill_to_decode(instance_name);
    }
  }
  
  std::unique_lock<std::shared_mutex> inst_lock(inst_mutex_);
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }
  instances_[instance_name].current_type = InstanceType::DECODE;
  LOG(INFO) << "Flip prefill to decode, instance name : " << instance_name;
}

// flip all models
void InstanceMgr::flip_decode_to_prefill(std::string& instance_name) {
  {
    std::shared_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
    // Flip in all managers
    for (auto& pair : model_instance_mgrs_) {
      pair.second->flip_decode_to_prefill(instance_name);
    }
  }
  
  std::unique_lock<std::shared_mutex> inst_lock(inst_mutex_);
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }
  instances_[instance_name].current_type = InstanceType::PREFILL;
  LOG(INFO) << "Flip decode to prefill, instance name : " << instance_name;
}

TimePredictor& InstanceMgr::get_time_predictor(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(time_predictor_mutex_);

  auto it = time_predictors_.find(instance_name);
  if (it == time_predictors_.end()) {
    LOG(FATAL) << "Find TimePredictor failed, instance name : "
               << instance_name;
  }
  return it->second;
}

double InstanceMgr::predict_ttft(const std::string& instance_name,
                                  const std::string& model_id,
                                  int32_t token_count) {
  std::lock_guard<std::mutex> lock(time_predictor_mutex_);
  auto it = time_predictors_.find(instance_name);
  if (it != time_predictors_.end()) {
    double ttft = it->second.predict_ttft(model_id, token_count);
    if (ttft > 0) {
      return ttft;
    }
  }
  // Fallback: ttft_ms = 0.0678 * tokens + 19.63 (fitted from profiling)
  return static_cast<double>(token_count) * 0.0964 + 67.27;
}

double InstanceMgr::predict_ttft_any_instance(const std::string& model_id,
                                               int32_t token_count) {
  std::lock_guard<std::mutex> lock(time_predictor_mutex_);
  for (auto& [instance_name, predictor] : time_predictors_) {
    double ttft = predictor.predict_ttft(model_id, token_count);
    if (ttft > 0) {
      return ttft;
    }
  }
  // Fallback: ttft_ms = 0.0678 * tokens + 19.63 (fitted from profiling)
  return static_cast<double>(token_count) * 0.0964 + 67.27;
}

void InstanceMgr::send_model_sleep(const std::string& instance_name,
                                   const std::string& model_id) {
  if (instance_name.empty() || instance_name == "all") {
    LOG(ERROR) << "Only support fixed instance_name for model trigger now.";
    return;
  }

  auto model_mgr = get_model_instance_mgr(model_id);

  std::shared_ptr<brpc::Channel> channel = get_channel(instance_name);

  if (model_mgr->send_model_sleep(instance_name, channel)) {
    LOG(INFO) << "Model " << model_id << " on " << instance_name
              << " sleep successful. Memory freed will be reflected in next heartbeat.";

    bool instance_idle = false;
    if (count_active_models_on_instance(instance_name) == 0) {
      std::unique_lock<std::shared_mutex> lock(tag_mutex_);
      instance_tag_map_[instance_name] = InstanceTag::NONE;
      LOG(INFO) << "Tag change: Reset tag for " << instance_name << " to NONE (no awake models)";
      refresh_elastic_decode_slot_locked(model_id);
      instance_idle = true;
    } else {
      std::unique_lock<std::shared_mutex> lock(tag_mutex_);
      refresh_elastic_decode_slot_locked(model_id);
    }
    if (instance_idle) {
      // Notify steady pool reclaim waiters.
      {
        std::lock_guard<std::mutex> lock2(instance_freed_mutex_);
      }
      instance_freed_cv_.notify_all();
    }
  }
}

bool InstanceMgr::send_model_wakeup(const std::string& instance_name,
                                    const std::string& model_id,
                                    bool memory_increased_in_advance) {

  if (instance_name.empty() || instance_name == "all") {
    LOG(ERROR) << "Only support fixed instance_name for model trigger now.";
    return false;
  }

  auto model_mgr = get_model_instance_mgr(model_id);

  std::shared_ptr<brpc::Channel> channel = get_channel(instance_name);

  // Try to find a D2D source instance (also acquires D2D lock if found)
  auto d2d_info = find_d2d_source(model_id, instance_name);

  bool wakeup_success = false;
  if (d2d_info.has_value()) {
    // Use D2D wakeup (mooncake connections already established at fork_master time)
    wakeup_success = model_mgr->send_model_wakeup_d2d(instance_name, channel, d2d_info.value());

    // Release D2D lock on the source instance (whether success or failure)
    model_mgr->release_d2d_lock(d2d_info->source_instance_name);

    if (!wakeup_success) {
      LOG(WARNING) << "D2D wakeup failed for model " << model_id
                   << " on " << instance_name << ", falling back to H2D";
      // D2D failure sets state to SLEEP, reset to ALLOCATED for H2D fallback
      model_mgr->set_model_state(instance_name, ModelState::ALLOCATED);
      // Fallback to H2D
      wakeup_success = model_mgr->send_model_wakeup(instance_name, channel);
    }
  } else {
    // Use H2D wakeup
    wakeup_success = model_mgr->send_model_wakeup(instance_name, channel);
  }

  if (wakeup_success) {
    LOG(INFO) << "Model " << model_id << " wakeup successful on " << instance_name
              << ". Memory usage will be reflected in next heartbeat.";
    // Notify cold elastic path waiters.
    model_wakeup_cv_.notify_all();
  } else {
    LOG(ERROR) << "Failed to wakeup model " << model_id
               << " on " << instance_name;
    // Roll back allocation state on failure.
    model_mgr->set_model_state(instance_name, ModelState::SLEEP);
    if (count_active_models_on_instance(instance_name) == 0) {
      std::unique_lock<std::shared_mutex> lock(tag_mutex_);
      instance_tag_map_[instance_name] = InstanceTag::NONE;
      LOG(INFO) << "Tag change:  Reset tag for " << instance_name
                << " to NONE (wakeup failed, no awake models)";
      refresh_elastic_decode_slot_locked(model_id);
    } else {
      std::unique_lock<std::shared_mutex> lock(tag_mutex_);
      refresh_elastic_decode_slot_locked(model_id);
    }
    {
      std::lock_guard<std::mutex> lock(allocation_mutex_);
      elastic_occupied_instances_.erase(instance_name);
    }
    // Notify steady pool reclaim waiters that an instance may be freed.
    {
      std::lock_guard<std::mutex> lock(instance_freed_mutex_);
    }
    instance_freed_cv_.notify_all();
  }
  return wakeup_success;
}

void InstanceMgr::init_model_memory_specs() {
    // Memory spec per GPU (GB) = weight_bytes / TP / (1024^3) + 5.0 (KV cache + overhead)
    // Model parameters sourced from janus_data/model_config.py

    // Qwen3-0.6B: 0.6B params × bf16 = 1.2 GB, TP=1, per-GPU = 1.2 + 5 = 6.2 GB
    model_memory_specs_["Qwen3-0.6B"] = 6.2;
    // Qwen3-1.7B: 1.7B params × bf16 = 3.4 GB, TP=1, per-GPU = 3.4 + 5 = 8.4 GB
    model_memory_specs_["Qwen3-1.7B"] = 8.4;
    // Qwen2.5-3B: 3.1B params × bf16 = 6.2 GB, TP=1, per-GPU = 6.2 + 5 = 11.2 GB
    model_memory_specs_["Qwen2.5-3B"] = 11.2;
    // Qwen3-4B: 4.0B params × bf16 = 8.0 GB, TP=1, per-GPU = 8.0 + 5 = 13.0 GB
    model_memory_specs_["Qwen3-4B"] = 13.0;
    // Qwen2-7B: 7.6B params × bf16 = 15.2 GB, TP=1, per-GPU = 15.2 + 5 = 20.2 GB
    model_memory_specs_["Qwen2-7B"] = 20.2;
    // Qwen3-8B: 8.2B params × bf16 = 16.4 GB, TP=1, per-GPU = 16.4 + 5 = 21.4 GB
    model_memory_specs_["Qwen3-8B"] = 21.4;
    // Qwen2.5-14B: 14.7B params × bf16 = 29.4 GB, TP=2, per-GPU = 14.7 + 5 = 19.7 GB
    model_memory_specs_["Qwen2.5-14B"] = 19.7;
    // Qwen3-32B: 32.8B params × bf16 = 65.6 GB, TP=2, per-GPU = 32.8 + 5 = 37.8 GB
    model_memory_specs_["Qwen3-32B"] = 37.8;
    // DeepSeek-V3.2: 671B params × FP8 = 671 GB, TP=16, per-GPU = 41.9 + 5 = 46.9 GB
    model_memory_specs_["DeepSeek-V3.2"] = 46.9;
    // GLM-4.5-Air: 112B params × FP8 = 112 GB, TP=16, per-GPU = 7.0 + 5 = 12.0 GB
    model_memory_specs_["GLM-4.5-Air"] = 12.0;
}

void InstanceMgr::init_model_resource_coefficients() {
  gpu_hw_spec_.hbm_per_gpu_gb = FLAGS_gpu_hbm_per_gpu_gb;
  gpu_hw_spec_.compute_sm_per_gpu = FLAGS_gpu_compute_sm_per_gpu;
  gpu_hw_spec_.bandwidth_per_gpu = FLAGS_gpu_bandwidth_per_gpu;

  // Load alias-to-real model mapping for GP lookup
  if (!FLAGS_model_alias_map_path.empty()) {
    std::ifstream alias_file(FLAGS_model_alias_map_path);
    if (alias_file.is_open()) {
      try {
        nlohmann::json alias_data;
        alias_file >> alias_data;
        for (auto it = alias_data.begin(); it != alias_data.end(); ++it) {
          alias_to_real_model_[it.key()] = it.value().get<std::string>();
        }
        LOG(INFO) << "Loaded " << alias_to_real_model_.size()
                  << " alias-to-real model mappings from "
                  << FLAGS_model_alias_map_path;
      } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to parse model alias map: " << e.what();
      }
    } else {
      LOG(ERROR) << "Failed to open model alias map file: "
                 << FLAGS_model_alias_map_path;
    }
  }

  // Load GP models from JSON files if paths are provided
  if (!FLAGS_gp_steady_data_path.empty()) {
    load_gp_steady_models(FLAGS_gp_steady_data_path);
  }
  if (!FLAGS_gp_dynamic_data_path.empty()) {
    load_gp_dynamic_models(FLAGS_gp_dynamic_data_path);
  }

  // Fallback: hardcoded linear models for models without GP data
  if (model_resource_models_.find("Qwen3-8B") == model_resource_models_.end()) {
    model_resource_models_["Qwen3-8B"] = std::make_unique<LinearResourceModel>(
        /*hbm_a=*/0.001, /*hbm_b=*/20.0,
        /*compute_a=*/0.0005, /*compute_b=*/0.0);
  }
  if (model_resource_models_.find("Qwen2-7B") == model_resource_models_.end()) {
    model_resource_models_["Qwen2-7B"] = std::make_unique<LinearResourceModel>(
        /*hbm_a=*/0.001, /*hbm_b=*/20.0,
        /*compute_a=*/0.0005, /*compute_b=*/0.0);
  }

  LOG(INFO) << "Initialized model resource models for "
            << model_resource_models_.size() << " steady models, "
            << dynamic_resource_models_.size() << " dynamic models, "
            << "GPU HBM=" << gpu_hw_spec_.hbm_per_gpu_gb << "GB, "
            << "GPU compute SM=" << gpu_hw_spec_.compute_sm_per_gpu << ", "
            << "GPU bandwidth=" << gpu_hw_spec_.bandwidth_per_gpu;
}

static std::unique_ptr<GaussianProcess> parse_gp_from_json(
    const nlohmann::json& j) {
  const auto& X_arr = j.at("X");
  const auto& y_arr = j.at("y");
  const auto& ls_arr = j.at("lengthscales");
  double signal_var = j.at("signal_variance").get<double>();
  double noise_var = j.at("noise_variance").get<double>();
  // Optional constant prior mean (MLE-fitted GP data, see fit_gp_mle.py).
  // Defaults to 0 for legacy zero-mean GP data files.
  double prior_mean = j.value("y_mean", 0.0);

  int n = X_arr.size();
  int d = ls_arr.size();

  Eigen::MatrixXd X(n, d);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < d; ++k) {
      X(i, k) = X_arr[i][k].get<double>();
    }
  }

  Eigen::VectorXd y(n);
  for (int i = 0; i < n; ++i) {
    y(i) = y_arr[i].get<double>();
  }

  Eigen::VectorXd ls(d);
  for (int i = 0; i < d; ++i) {
    ls(i) = ls_arr[i].get<double>();
  }

  return std::make_unique<GaussianProcess>(X, y, ls, signal_var, noise_var,
                                           prior_mean);
}

void InstanceMgr::load_gp_steady_models(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open GP steady data file: " << path;
    return;
  }

  nlohmann::json data;
  try {
    file >> data;
  } catch (const nlohmann::json::parse_error& e) {
    LOG(ERROR) << "Failed to parse GP steady data JSON: " << e.what();
    return;
  }

  for (auto it = data.begin(); it != data.end(); ++it) {
    const std::string& model_id = it.key();
    const auto& model_data = it.value();

    try {
      auto gp_hbm = parse_gp_from_json(model_data.at("hbm"));
      auto gp_compute = parse_gp_from_json(model_data.at("compute"));
      auto gp_bandwidth = parse_gp_from_json(model_data.at("bandwidth"));

      model_resource_models_[model_id] =
          std::make_unique<GPSteadyResourceModel>(
              std::move(gp_hbm), std::move(gp_compute),
              std::move(gp_bandwidth));

      LOG(INFO) << "Loaded GP steady resource model for " << model_id;
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to load GP steady model for " << model_id
                 << ": " << e.what();
    }
  }
}

void InstanceMgr::load_gp_dynamic_models(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open GP dynamic data file: " << path;
    return;
  }

  nlohmann::json data;
  try {
    file >> data;
  } catch (const nlohmann::json::parse_error& e) {
    LOG(ERROR) << "Failed to parse GP dynamic data JSON: " << e.what();
    return;
  }

  for (auto it = data.begin(); it != data.end(); ++it) {
    const std::string& model_id = it.key();
    const auto& model_data = it.value();

    try {
      auto gp_slo = parse_gp_from_json(model_data);

      dynamic_resource_models_[model_id] =
          std::make_unique<GPDynamicResourceModel>(std::move(gp_slo));

      LOG(INFO) << "Loaded GP dynamic resource model for " << model_id;
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to load GP dynamic model for " << model_id
                 << ": " << e.what();
    }
  }
}

// TODO: support dynamic instance memory specs, rather than hardcoded.
double InstanceMgr::get_model_memory_size(const std::string& model_id) {
    if (model_memory_specs_.count(model_id)) {
        return model_memory_specs_[model_id];
    }
    // Resolve alias to real model
    const std::string& real_model_id = resolve_gp_model_id(model_id);
    if (real_model_id != model_id && model_memory_specs_.count(real_model_id)) {
        return model_memory_specs_[real_model_id];
    }
    LOG(WARNING) << "Unknown model ID for memory spec: " << model_id << ", using default 20GB";
    return 20.0;
}

bool InstanceMgr::is_model_waking_up(const std::string& model_id) {
  auto model_mgr = get_model_instance_mgr(model_id);
  return model_mgr->is_model_waking_up();
}

std::vector<std::string> InstanceMgr::get_awake_instances(const std::string& model_id) {
  auto model_mgr = get_model_instance_mgr(model_id);
  return model_mgr->get_awake_instances();
}

std::vector<std::string> InstanceMgr::get_awake_prefill_instances(const std::string& model_id) {
  auto model_mgr = get_model_instance_mgr(model_id);
  auto all_instances = model_mgr->get_awake_instances();

  std::vector<std::string> filtered;
  for (const auto& inst : all_instances) {
    if (get_instance_tag(inst) != InstanceTag::DECODE) {
      filtered.push_back(inst);
    }
  }
  return filtered;
}

void InstanceMgr::update_model_heat(const std::string& model_id,
                                    int64_t token_count,
                                    int64_t input_len) {
  auto model_mgr = get_model_instance_mgr(model_id);
  model_mgr->update_model_heat(token_count, input_len);
}

int32_t InstanceMgr::get_wakeup_count(const std::string& model_id) {
  auto model_mgr = get_model_instance_mgr(model_id);
  return model_mgr->get_wakeup_count();
}


bool InstanceMgr::wait_for_model_drain(const std::string& instance_name,
                                        const std::string& model_id) {
  std::unique_lock<std::mutex> lock(request_metrics_mutex_);
  auto instance_it = request_metrics_.find(instance_name);
  if (instance_it == request_metrics_.end()) {
    LOG(ERROR) << "wait_for_model_drain: no request metrics for " << instance_name;
    return false;
  }
  auto model_it = instance_it->second.model_metrics.find(model_id);
  if (model_it == instance_it->second.model_metrics.end()) {
    LOG(ERROR) << "wait_for_model_drain: no model metrics for " << model_id
               << " on " << instance_name;
    return false;
  }
  auto& metrics = model_it->second;
  metrics.cv_idle.wait(lock, [&metrics]() {
    return metrics.prefill_request_num == 0 && metrics.decode_request_num == 0;
  });
  return true;
}

bool InstanceMgr::should_accept_scaling_plan(
    const ScalingPlan& new_plan) const {
  // No previous plan — always accept.
  if (current_scaling_plan_.empty()) return true;

  const auto& now_plan = current_scaling_plan_;

  // Rule 1: Model set changed (new model added or existing model removed).
  for (const auto& [model, _] : new_plan) {
    if (now_plan.find(model) == now_plan.end()) return true;
  }
  for (const auto& [model, _] : now_plan) {
    if (new_plan.find(model) == new_plan.end()) return true;
  }

  // Rule 2: Any model has a large heat change (>50%) or large instance
  //         change (>=3) — accept immediately.
  for (const auto& [model, new_entry] : new_plan) {
    const auto& now_entry = now_plan.at(model);
    // Heat change check
    int64_t heat_base = std::max(now_entry.heat, new_entry.heat);
    if (heat_base > 0) {
      int64_t heat_delta = std::abs(new_entry.heat - now_entry.heat);
      double heat_ratio = static_cast<double>(heat_delta) / heat_base;
      if (heat_ratio > 0.5) return true;
    }
    // Instance change check
    int32_t inst_delta = std::abs(new_entry.gpu_allocated - now_entry.gpu_allocated);
    if (inst_delta >= 3) return true;
  }

  auto elapsed = std::chrono::steady_clock::now() - last_plan_change_time_;
  double elapsed_sec =
      std::chrono::duration<double>(elapsed).count();

  // Rule 3: All models have tiny changes (heat change <=20% AND instance
  //         change <=1) and the last plan change was <5s ago — reject.
  {
    bool all_tiny = true;
    for (const auto& [model, new_entry] : new_plan) {
      const auto& now_entry = now_plan.at(model);
      int64_t heat_base = std::max(now_entry.heat, new_entry.heat);
      double heat_ratio = (heat_base > 0)
          ? static_cast<double>(std::abs(new_entry.heat - now_entry.heat)) / heat_base
          : 0.0;
      int32_t inst_delta = std::abs(new_entry.gpu_allocated - now_entry.gpu_allocated);
      if (!(heat_ratio <= 0.2 && inst_delta <= 1)) {
        all_tiny = false;
        break;
      }
    }
    if (all_tiny && elapsed_sec < 5.0) {
      // LOG(INFO) << "Anti-jitter: reject (all changes tiny, elapsed="
      //           << elapsed_sec << "s < 5s)";
      return false;
    }
  }

  return true;



  // Rule 4: Direction reversal detection via dot product.
  //         If dot(now - last, new - now) < 0 and elapsed < 5s — reject.
  if (!last_scaling_plan_.empty() && elapsed_sec < 5.0) {
    // Build direction vectors over the union of all models across the three
    // plans. Missing models in a plan are treated as 0 instances.
    std::unordered_set<std::string> all_models;
    for (const auto& [m, _] : last_scaling_plan_) all_models.insert(m);
    for (const auto& [m, _] : now_plan) all_models.insert(m);
    for (const auto& [m, _] : new_plan) all_models.insert(m);

    auto get_gpu = [](const ScalingPlan& plan,
                      const std::string& m) -> int32_t {
      auto it = plan.find(m);
      return (it != plan.end()) ? it->second.gpu_allocated : 0;
    };

    double dot = 0.0;
    for (const auto& m : all_models) {
      int32_t last_v = get_gpu(last_scaling_plan_, m);
      int32_t now_v = get_gpu(now_plan, m);
      int32_t new_v = get_gpu(new_plan, m);
      double d_prev = static_cast<double>(now_v - last_v);   // last→now
      double d_next = static_cast<double>(new_v - now_v);    // now→new
      dot += d_prev * d_next;
    }
    if (dot < 0.0) {
      // LOG(INFO) << "Anti-jitter: reject (direction reversal, dot="
      //           << dot << ", elapsed=" << elapsed_sec << "s < 5s)";
      return false;
    }
  }

  // Rule 5: Accept by default.
  return true;
}

void InstanceMgr::dynamic_part_auto_scaling() {
  std::unique_lock<std::mutex> alloc_lock(allocation_mutex_);
  dynamic_part_auto_scaling_impl();
}

bool InstanceMgr::try_dynamic_part_auto_scaling() {
  std::unique_lock<std::mutex> alloc_lock(allocation_mutex_, std::try_to_lock);
  if (!alloc_lock.owns_lock()) {
    return false;
  }
  dynamic_part_auto_scaling_impl();
  return true;
}

void InstanceMgr::request_cold_elastic_wakeup(const std::string& model_id) {
  // Signal the auto-scaling thread to run immediately.
  {
    std::lock_guard<std::mutex> lock(scaling_trigger_mutex_);
    scaling_requested_ = true;
  }
  scaling_trigger_cv_.notify_one();

  // Wait until model has at least 1 PREFILL + 1 DECODE instance in WAKEUP
  // state (or timeout). A single wakeup_count > 0 is insufficient because
  // the first instance to finish wakeup may be DECODE-tagged, leaving the
  // prefill list empty and causing a spurious 503.
  static constexpr int kColdWakeupTimeoutSeconds = 10;
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::seconds(kColdWakeupTimeoutSeconds);
  std::unique_lock<std::mutex> lock(model_wakeup_wait_mutex_);
  model_wakeup_cv_.wait_until(lock, deadline, [this, &model_id] {
    return (!get_awake_prefill_instances(model_id).empty() &&
            !get_awake_decode_instances(model_id).empty()) ||
           exited_;
  });
}

void InstanceMgr::dynamic_part_auto_scaling_impl() {
  if (options_.disable_elastic_pool()) {
    return;
  }

  int32_t total_gpus = total_available_gpus_.load();
  int32_t raw_budget = std::max(0, total_gpus - steady_needed_gpus());

  // 1. Compute per-model GPU targets (elastic pool models only)
  struct ScalingTarget {
    std::string model_id;
    int64_t heat;
    int32_t gpu_target;
    int32_t gpu_allocated;
    std::shared_ptr<ModelInstanceMgr> mgr;
  };
  std::vector<ScalingTarget> targets;
  int32_t total_elastic_alloc = 0;

  {
    std::shared_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
    for (const auto& [id, mgr] : model_instance_mgrs_) {
      // Only include elastic pool models
      auto pool_it = model_pool_assignments_.find(id);
      if (pool_it == model_pool_assignments_.end() ||
          pool_it->second != PoolType::ELASTIC) {
        continue;
      }
      total_elastic_alloc += mgr->get_allocation_count();

      ScalingTarget t;
      t.model_id = id;
      t.mgr = mgr;
      t.heat = mgr->get_model_heat();

      if (t.heat == 0) {
        t.gpu_target = 0;
      } else {
        double avg_rate = mgr->get_avg_token_rate(3);
        t.gpu_target = compute_elastic_gpu_target_from_rate(avg_rate);
      }

      t.gpu_allocated = 0;
      targets.push_back(std::move(t));
    }
  }

  // Budget: subtract steady reservation + in-flight DRAINING instances.
  // DRAINING instances still occupy GPUs but are not counted in allocation_count.
  int32_t in_flight = static_cast<int32_t>(elastic_occupied_instances_.size()) -
                      total_elastic_alloc;
  int32_t budget = std::max(0, raw_budget - std::max(0, in_flight));

  // 2. Budget constraint with proportional scaling.
  // gpu_target = number of PREFILL instances needed.  Each active model
  // also requires exactly 1 DECODE instance.  Algorithm:
  //   (a) Reserve 1 DECODE GPU per active model.
  //   (b) Distribute the remaining budget as PREFILL instances.
  //   (c) If a model gets 0 PREFILL, revoke its DECODE reservation and
  //       redistribute the freed GPU.  Iterate until stable.
  std::vector<bool> eligible(targets.size(), false);
  int32_t num_eligible = 0;
  for (size_t i = 0; i < targets.size(); ++i) {
    if (targets[i].gpu_target > 0) {
      eligible[i] = true;
      num_eligible++;
    }
  }

  bool converged = false;
  while (!converged && num_eligible > 0) {
    converged = true;

    // Reserve 1 DECODE GPU per eligible model.
    int32_t prefill_budget = budget - num_eligible;

    if (prefill_budget <= 0) {
      // Cannot even reserve 1 DECODE per model — demote coldest.
      size_t coldest = 0;
      int64_t min_heat = std::numeric_limits<int64_t>::max();
      for (size_t i = 0; i < targets.size(); ++i) {
        if (eligible[i] && targets[i].heat < min_heat) {
          min_heat = targets[i].heat;
          coldest = i;
        }
      }
      LOG(INFO) << "Decode reservation exhausted budget: demoting model "
                << targets[coldest].model_id
                << " (heat=" << targets[coldest].heat
                << ") — budget=" << budget;
      eligible[coldest] = false;
      targets[coldest].gpu_allocated = 0;
      num_eligible--;
      converged = false;
      continue;
    }

    // Compute total PREFILL demand among eligible models.
    int32_t prefill_demand = 0;
    for (size_t i = 0; i < targets.size(); ++i) {
      if (eligible[i]) prefill_demand += targets[i].gpu_target;
    }

    if (prefill_demand <= prefill_budget) {
      // Full allocation: every eligible model gets gpu_target P + 1 D.
      for (size_t i = 0; i < targets.size(); ++i) {
        targets[i].gpu_allocated = eligible[i]
            ? targets[i].gpu_target + 1 : 0;
      }
      break;
    }

    // Proportional scaling of PREFILL instances (floor, min 1P guarantee).
    double ratio = static_cast<double>(prefill_budget) / prefill_demand;
    int32_t sum_prefill = 0;
    for (size_t i = 0; i < targets.size(); ++i) {
      if (!eligible[i]) {
        targets[i].gpu_allocated = 0;
        continue;
      }
      int32_t p = std::max(1, static_cast<int32_t>(
          std::floor(targets[i].gpu_target * ratio)));
      targets[i].gpu_allocated = p;          // prefill only; decode added later
      sum_prefill += p;
    }

    // Distribute leftover PREFILL GPUs via Largest Remainder Method.
    if (sum_prefill < prefill_budget) {
      int32_t prefill_remainder = prefill_budget - sum_prefill;
      struct RemainderCandidate {
        size_t index;
        double fractional;
        int64_t heat;
      };
      std::vector<RemainderCandidate> candidates;
      for (size_t i = 0; i < targets.size(); ++i) {
        if (!eligible[i] || targets[i].gpu_target == 0) continue;
        double raw = targets[i].gpu_target * ratio;
        int32_t floored = std::max(1, static_cast<int32_t>(std::floor(raw)));
        if (floored > static_cast<int32_t>(std::floor(raw)) &&
            targets[i].gpu_allocated == floored) continue;
        double frac = raw - targets[i].gpu_allocated;
        candidates.push_back({i, frac, targets[i].heat});
      }
      std::sort(candidates.begin(), candidates.end(),
                [](const RemainderCandidate& a, const RemainderCandidate& b) {
                  if (a.fractional != b.fractional)
                    return a.fractional > b.fractional;
                  return a.heat < b.heat;
                });
      for (int32_t r = 0;
           r < prefill_remainder &&
           r < static_cast<int32_t>(candidates.size()); ++r) {
        targets[candidates[r].index].gpu_allocated++;
        sum_prefill++;
      }
    }

    // Build index list sorted by heat ascending (coldest first)
    // for trim and demotion phases.
    std::vector<size_t> by_heat;
    for (size_t i = 0; i < targets.size(); ++i) {
      if (eligible[i]) by_heat.push_back(i);
    }
    std::sort(by_heat.begin(), by_heat.end(),
              [&targets](size_t a, size_t b) {
                return targets[a].heat < targets[b].heat;
              });

    // Trim overshoot from coldest models (never below 1P).
    while (sum_prefill > prefill_budget) {
      bool trimmed = false;
      for (size_t idx : by_heat) {
        if (targets[idx].gpu_allocated > 1) {
          targets[idx].gpu_allocated--;
          sum_prefill--;
          trimmed = true;
          if (sum_prefill <= prefill_budget) break;
        }
      }
      if (!trimmed) break;
    }

    // Still over budget (all eligible at 1P) → demote coldest entirely.
    while (sum_prefill > prefill_budget) {
      bool demoted = false;
      for (size_t idx : by_heat) {
        if (!eligible[idx]) continue;
        LOG(INFO) << "Prefill budget exceeded: demoting model "
                  << targets[idx].model_id << " (heat=" << targets[idx].heat
                  << ", prefill=" << targets[idx].gpu_allocated
                  << ") — budget=" << budget;
        sum_prefill -= targets[idx].gpu_allocated;
        targets[idx].gpu_allocated = 0;
        eligible[idx] = false;
        num_eligible--;
        converged = false;
        demoted = true;
        if (sum_prefill <= prefill_budget) break;
      }
      if (!demoted) break;
    }

    if (converged) {
      // Add 1 DECODE instance per eligible model.
      for (size_t i = 0; i < targets.size(); ++i) {
        if (eligible[i]) targets[i].gpu_allocated += 1;
      }
    }
  }

  if (num_eligible == 0) {
    for (auto& t : targets) t.gpu_allocated = 0;
  }

  for (auto& t : targets) {
    LOG(INFO) << "Scaling plan: model=" << t.model_id
              << " pool=ELASTIC"
              << " heat=" << t.heat
              << " gpu_target=" << t.gpu_target
              << " gpu_allocated=" << t.gpu_allocated
              << " budget=" << budget;
  }

  // Anti-jitter check: compare new plan against previous plans.
  // Use PREFILL count only (exclude the fixed DECODE instance) so that
  // the jitter detector tracks the variable part of the allocation.
  ScalingPlan new_plan;
  for (const auto& t : targets) {
    int32_t prefill_count = (t.gpu_allocated > 1) ? t.gpu_allocated - 1 : 0;
    new_plan[t.model_id] = {prefill_count, t.heat};
  }
  if (!should_accept_scaling_plan(new_plan)) {
    return;
  }
  // Accepted — rotate plan history.
  last_scaling_plan_ = std::move(current_scaling_plan_);
  current_scaling_plan_ = std::move(new_plan);
  last_plan_change_time_ = std::chrono::steady_clock::now();

  // 3. Overlapped scale-up/scale-down
  //
  // Phase A: Build scale-down candidates and scale-up needs (round-robin)
  struct ScaleUpNeed {
    size_t target_idx;
    std::string model_id;
    uint64_t model_size;
    bool matched = false;
  };
  struct ScaleDownCandidate {
    std::string instance_name;
    size_t target_idx;
    std::string model_id;
    bool matched = false;
  };

  std::vector<ScaleDownCandidate> scale_down_candidates;
  for (size_t i = 0; i < targets.size(); ++i) {
    // Use allocation_count (ALLOCATED + WAKEUP) to treat in-flight async
    // wakeups as working instances, preventing unnecessary scale-down.
    int32_t current_alloc = targets[i].mgr->get_allocation_count();
    int32_t excess = current_alloc - targets[i].gpu_allocated;
    if (excess <= 0) continue;
    auto unlocked = targets[i].mgr->get_unlocked_instances();
    int32_t added = 0;
    for (size_t j = 0; j < unlocked.size() && added < excess; ++j) {
      // Protect DECODE instances from scale-down unless the model is being
      // fully removed (gpu_allocated == 0).
      if (targets[i].gpu_allocated > 0 &&
          get_instance_tag(unlocked[j]) == InstanceTag::DECODE) {
        continue;
      }
      scale_down_candidates.push_back(
          {unlocked[j], i, targets[i].model_id, false});
      ++added;
    }
  }

  // Build scale-up needs in round-robin order for fairness
  // E.g., A needs +5, B needs +4, C needs +3 → A,B,C,A,B,C,A,B,C,A,B,A
  struct ModelDeficit {
    size_t target_idx;
    int32_t remaining;
    std::string model_id;
    uint64_t model_size;
  };
  std::vector<ModelDeficit> model_deficits;
  for (size_t i = 0; i < targets.size(); ++i) {
    int32_t current_alloc = targets[i].mgr->get_allocation_count();
    int32_t deficit = targets[i].gpu_allocated - current_alloc;
    if (deficit > 0) {
      model_deficits.push_back({i, deficit, targets[i].model_id,
                                get_model_size_bytes(targets[i].model_id)});
    }
  }
  std::vector<ScaleUpNeed> scale_up_needs;
  while (!model_deficits.empty()) {
    for (auto it = model_deficits.begin(); it != model_deficits.end(); ) {
      scale_up_needs.push_back(
          {it->target_idx, it->model_id, it->model_size, false});
      if (--it->remaining <= 0) {
        it = model_deficits.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Phase B: Greedy matching — pair scale-up needs with scale-down candidates
  // Best fit: smallest free_bytes >= model_size + abundance (can hold both models at once)
  struct OverlapMatch {
    size_t up_idx;
    size_t down_idx;
  };
  std::vector<OverlapMatch> overlapped_matches;

  const uint64_t overlap_abundance_bytes =
      static_cast<uint64_t>(FLAGS_overlap_abundance_gb * 1024 * 1024 * 1024);

  for (size_t u = 0; u < scale_up_needs.size(); ++u) {
    auto& up = scale_up_needs[u];
    int best_idx = -1;
    uint64_t best_free = UINT64_MAX;
    const uint64_t required_bytes = up.model_size + overlap_abundance_bytes;
    for (size_t k = 0; k < scale_down_candidates.size(); ++k) {
      if (scale_down_candidates[k].matched) continue;
      uint64_t free_bytes =
          get_instance_free_bytes(scale_down_candidates[k].instance_name);
      if (free_bytes >= required_bytes && free_bytes < best_free) {
        best_free = free_bytes;
        best_idx = static_cast<int>(k);
      }
    }
    if (best_idx >= 0) {
      overlapped_matches.push_back({u, static_cast<size_t>(best_idx)});
      scale_down_candidates[best_idx].matched = true;
      scale_up_needs[u].matched = true;
    }
  }

  // Phase C: Execute overlapped matches (wake new model while draining old)
  for (auto& m : overlapped_matches) {
    auto& up = scale_up_needs[m.up_idx];
    auto& down = scale_down_candidates[m.down_idx];

    LOG(INFO) << "Overlapped scale: wake " << up.model_id
              << " while draining " << down.model_id
              << " on " << down.instance_name;

    // Mark old model as DRAINING (stops routing new requests to it)
    targets[down.target_idx].mgr->set_model_state(
        down.instance_name, ModelState::DRAINING);

    // Allocate + wake new model on same instance (state bookkeeping synchronous)
    targets[up.target_idx].mgr->set_model_state(
        down.instance_name, ModelState::ALLOCATED);
    deduct_free_pages(down.instance_name,
                      get_model_size_bytes(up.model_id));
    {
      std::unique_lock<std::shared_mutex> lock(tag_mutex_);
      auto new_tag = determine_elastic_tag_locked(up.model_id);
      instance_tag_map_[down.instance_name] = new_tag;
      LOG(INFO) << "Tag change: instance " << down.instance_name
                << " -> " << instance_tag_name(new_tag)
                << " (overlapped scale-up for model " << up.model_id << ")";
    }
    // Async: wakeup new model (HTTP call in separate thread).
    std::string wake_inst = down.instance_name;
    std::string wake_model = up.model_id;
    std::thread([this, wake_inst, wake_model]() {
      send_model_wakeup(wake_inst, wake_model, true);
    }).detach();

    // Async: drain + sleep old model
    // Instance stays in elastic_occupied_instances_ (new model is using it)
    std::string inst = down.instance_name;
    std::string old_model = down.model_id;
    std::thread([this, inst, old_model]() {
      if (!wait_for_model_drain(inst, old_model)) return;
      auto mgr = get_model_instance_mgr(old_model);
      if (mgr && !mgr->can_sleep(inst)) return;
      send_model_sleep(inst, old_model);
      promote_prefill_to_decode(old_model);
    }).detach();
  }

  // Phase D: Execute remaining scale-ups (on free instances, async wakeup)
  for (auto& t : targets) {
    int32_t current_alloc = t.mgr->get_allocation_count();
    int32_t deficit = t.gpu_allocated - current_alloc;
    if (deficit <= 0) continue;

    uint64_t model_size = get_model_size_bytes(t.model_id);

    std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
    for (const auto& [inst_name, _] : instances_) {
      if (deficit <= 0) break;
      if (get_instance_tag(inst_name) != InstanceTag::NONE) continue;
      if (!has_valid_xtensor_info(inst_name)) {
        LOG(WARNING) << "dynamic_part_auto_scaling: skip " << inst_name
                     << " for model " << t.model_id
                     << " (no valid xtensor info)";
        continue;
      }
      if (get_instance_free_bytes(inst_name) < model_size) {
        LOG(INFO) << "dynamic_part_auto_scaling: skip " << inst_name
                  << " for model " << t.model_id << " (not enough space: free="
                  << get_instance_free_bytes(inst_name)
                  << " need=" << model_size << ")";
        continue;
      }

      // Synchronous state bookkeeping.
      t.mgr->set_model_state(inst_name, ModelState::ALLOCATED);
      deduct_free_pages(inst_name, model_size);
      elastic_occupied_instances_.insert(inst_name);
      {
        std::unique_lock<std::shared_mutex> lock(tag_mutex_);
        auto new_tag = determine_elastic_tag_locked(t.model_id);
        instance_tag_map_[inst_name] = new_tag;
        LOG(INFO) << "Tag change: instance " << inst_name
                  << " -> " << instance_tag_name(new_tag)
                  << " (elastic scale-up for model " << t.model_id << ")";
      }
      // Async wakeup (HTTP call in separate thread).
      std::string wi = inst_name;
      std::string wm = t.model_id;
      std::thread([this, wi, wm]() {
        send_model_wakeup(wi, wm, true);
      }).detach();
      deficit--;
    }
  }

  // Phase E: Execute remaining scale-downs (async drain + sleep)
  for (auto& down : scale_down_candidates) {
    if (down.matched) continue;
    targets[down.target_idx].mgr->set_model_state(
        down.instance_name, ModelState::DRAINING);
    std::string inst = down.instance_name;
    std::string model = down.model_id;
    std::thread([this, inst, model]() {
      if (!wait_for_model_drain(inst, model)) return;
      auto mgr = get_model_instance_mgr(model);
      if (mgr && !mgr->can_sleep(inst)) return;
      send_model_sleep(inst, model);
      {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        elastic_occupied_instances_.erase(inst);
      }
      // Notify steady pool reclaim waiters that an instance may be freed.
      {
        std::lock_guard<std::mutex> lock(instance_freed_mutex_);
      }
      instance_freed_cv_.notify_all();
      promote_prefill_to_decode(model);
    }).detach();
  }

  // Cleanup: if heat → 0, remove from pool and drain+sleep remaining instances
  for (auto& t : targets) {
    if (t.gpu_allocated == 0) {
      model_pool_assignments_[t.model_id] = PoolType::NONE;
      {
        std::unique_lock<std::shared_mutex> lock(tag_mutex_);
        clear_elastic_decode_slot_locked(
            t.model_id, "model removed from ELASTIC pool");
      }
      LOG(INFO) << "Model " << t.model_id
                << " heat=0, removed from ELASTIC pool";

      // Drain+sleep all instances still active (WAKEUP or ALLOCATED).
      // Using get_active_instances() to also catch in-flight wakeups that
      // haven't completed yet, preventing leaked instances.
      auto awake_instances = t.mgr->get_active_instances();
      for (const auto& inst_name : awake_instances) {
        t.mgr->set_model_state(inst_name, ModelState::DRAINING);
        std::string inst = inst_name;
        std::string model = t.model_id;
        std::thread([this, inst, model]() {
          if (!wait_for_model_drain(inst, model)) return;
          auto mgr = get_model_instance_mgr(model);
          if (mgr && !mgr->can_sleep(inst)) return;
          send_model_sleep(inst, model);
          {
            std::lock_guard<std::mutex> lock(allocation_mutex_);
            elastic_occupied_instances_.erase(inst);
          }
          {
            std::lock_guard<std::mutex> lock(instance_freed_mutex_);
          }
          instance_freed_cv_.notify_all();
          promote_prefill_to_decode(model);
        }).detach();
      }
    }
  }
}

std::shared_ptr<ModelInstanceMgr> InstanceMgr::get_model_instance_mgr(const std::string& model_id) {
  std::shared_lock<std::shared_mutex> lock(model_instance_mgr_mutex_);
  auto it = model_instance_mgrs_.find(model_id);
  if (it != model_instance_mgrs_.end()) {
    return it->second;
  }
  LOG(ERROR) << "Model instance manager not found for model " << model_id;
  return nullptr;
}

bool InstanceMgr::has_valid_xtensor_info(const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
  auto it = instance_xtensor_infos_.find(instance_name);
  return it != instance_xtensor_infos_.end() && it->second.is_valid;
}

uint64_t InstanceMgr::get_model_size_bytes(const std::string& model_id) {
  // Resolve alias to real model for lookup
  const std::string& real_model_id = resolve_gp_model_id(model_id);

  // Priority 1: Get from any instance's xtensor info (try both alias and real)
  {
    std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
    for (const auto& [inst_name, info] : instance_xtensor_infos_) {
      if (!info.is_valid) continue;
      uint64_t size = info.get_model_size_bytes(model_id);
      if (size > 0) return size;
      if (real_model_id != model_id) {
        size = info.get_model_size_bytes(real_model_id);
        if (size > 0) return size;
      }
    }
  }

  // Priority 2: Fallback to model_memory_specs_ (try both alias and real)
  if (model_memory_specs_.count(model_id)) {
    return static_cast<uint64_t>(model_memory_specs_[model_id] * 1024 * 1024 * 1024);
  }
  if (real_model_id != model_id && model_memory_specs_.count(real_model_id)) {
    return static_cast<uint64_t>(model_memory_specs_[real_model_id] * 1024 * 1024 * 1024);
  }

  // Default fallback: 20GB
  LOG(WARNING) << "Unknown model size for " << model_id << ", using default 20GB";
  return 20ULL * 1024 * 1024 * 1024;
}

uint64_t InstanceMgr::get_instance_free_bytes(const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
  auto it = instance_xtensor_infos_.find(instance_name);
  if (it == instance_xtensor_infos_.end() || !it->second.is_valid) {
    return 0;  // No valid info, cannot allocate
  }
  return it->second.get_min_free_bytes();
}

bool InstanceMgr::has_enough_space_for_model(const std::string& instance_name,
                                              const std::string& model_id) {
  uint64_t free_bytes = get_instance_free_bytes(instance_name);
  uint64_t model_size = get_model_size_bytes(model_id);
  return free_bytes >= model_size;
}

void InstanceMgr::deduct_free_pages(const std::string& instance_name, uint64_t bytes) {
  std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
  auto it = instance_xtensor_infos_.find(instance_name);
  if (it == instance_xtensor_infos_.end() || !it->second.is_valid) return;

  uint64_t pages_to_deduct = (bytes + kXTensorPageSizeBytes - 1) / kXTensorPageSizeBytes;
  for (auto& pages : it->second.worker_free_phy_pages) {
    pages = (pages > pages_to_deduct) ? (pages - pages_to_deduct) : 0;
  }
}

void InstanceMgr::update_xtensor_info(
    const std::string& instance_name,
    const proto::XTensorHeartbeatInfo& xtensor_info) {
  std::lock_guard<std::mutex> lock(xtensor_info_mutex_);

  InstanceXTensorInfo info;
  info.is_valid = true;  // Mark as valid heartbeat data

  // Copy worker_free_phy_pages
  for (auto pages : xtensor_info.worker_free_phy_pages()) {
    info.worker_free_phy_pages.push_back(pages);
  }

  // Copy model_weight_segments
  for (const auto& [model_id, segment_list] : xtensor_info.model_weight_segments()) {
    std::vector<WeightSegment> segments;
    for (const auto& seg : segment_list.segments()) {
      segments.push_back({seg.offset(), seg.size()});
    }
    info.model_weight_segments[model_id] = std::move(segments);
  }

  // Preserve device_addrs and p2p_addrs from registration (not reported via heartbeat)
  {
    auto it = instance_xtensor_infos_.find(instance_name);
    if (it != instance_xtensor_infos_.end()) {
      info.device_addrs = it->second.device_addrs;
      info.p2p_addrs = it->second.p2p_addrs;
    }
  }

  instance_xtensor_infos_[instance_name] = std::move(info);
}

std::optional<InstanceXTensorInfo> InstanceMgr::get_instance_xtensor_info(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
  auto it = instance_xtensor_infos_.find(instance_name);
  if (it != instance_xtensor_infos_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::optional<D2DWakeupInfo> InstanceMgr::find_d2d_source(
    const std::string& model_id,
    const std::string& target_instance_name) {
  // Find instances that have the model in WAKEUP state
  auto model_mgr = get_model_instance_mgr(model_id);
  if (!model_mgr) {
    LOG(WARNING) << "Model manager not found for " << model_id;
    return std::nullopt;
  }

  // Atomically get awake instances and lock all of them
  auto locked_instances = model_mgr->get_awake_instances_and_lock();
  if (locked_instances.empty()) {
    LOG(INFO) << "No awake instances found for model " << model_id
              << ", will use H2D wakeup";
    return std::nullopt;
  }

  // Track which instance we select as source (to keep it locked)
  std::string selected_source;

  // Collect all eligible D2D source candidates, then pick the most loaded one.
  struct D2DCandidate {
    std::string instance_name;
    std::vector<std::string> p2p_addrs;
    std::vector<WeightSegment> weight_segments;
    int64_t epdt;  // estimated prefill done time (higher = more loaded)
  };
  std::vector<D2DCandidate> candidates;

  // Find all suitable source instances (not the target itself)
  for (const auto& source_instance_name : locked_instances) {
    if (source_instance_name == target_instance_name) {
      continue;
    }

    // Check if we have XTensor info for this source instance
    std::optional<InstanceXTensorInfo> xtensor_info;
    {
      std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
      auto it = instance_xtensor_infos_.find(source_instance_name);
      if (it != instance_xtensor_infos_.end() && it->second.is_valid) {
        xtensor_info = it->second;
      }
    }

    if (!xtensor_info.has_value()) {
      continue;
    }

    // Check if the source has weight segments for this model
    auto seg_it = xtensor_info->model_weight_segments.find(model_id);
    if (seg_it == xtensor_info->model_weight_segments.end() ||
        seg_it->second.empty()) {
      continue;
    }

    // Get P2P addresses for D2D weight transfer
    std::vector<std::string> p2p_addrs;
    if (!xtensor_info->p2p_addrs.empty()) {
      p2p_addrs = xtensor_info->p2p_addrs;
    } else {
      std::shared_lock<std::shared_mutex> lock(inst_mutex_);
      auto meta_it = instances_.find(source_instance_name);
      if (meta_it != instances_.end()) {
        p2p_addrs = meta_it->second.p2p_addrs;
      }
    }

    if (p2p_addrs.empty()) {
      continue;
    }

    // Eligible candidate — record with EPDT for load ranking
    int64_t epdt = get_estimated_prefill_done_time(source_instance_name);
    candidates.push_back({source_instance_name, std::move(p2p_addrs),
                          seg_it->second, epdt});
  }

  if (candidates.empty()) {
    model_mgr->release_d2d_locks(locked_instances);
    LOG(INFO) << "No suitable D2D source found for model " << model_id
              << ", will use H2D wakeup";
    return std::nullopt;
  }

  // Prefer the most heavily-loaded instance (highest EPDT) as D2D source.
  // This avoids choosing lightly-loaded instances that are more likely to be
  // drained/evicted.
  auto& best = *std::max_element(
      candidates.begin(), candidates.end(),
      [](const D2DCandidate& a, const D2DCandidate& b) {
        return a.epdt < b.epdt;
      });

  // Build D2DWakeupInfo from best candidate
  D2DWakeupInfo d2d_info;
  d2d_info.source_instance_name = best.instance_name;
  d2d_info.remote_addrs = best.p2p_addrs;
  for (size_t i = 0; i < best.p2p_addrs.size(); ++i) {
    d2d_info.src_weight_segments.push_back(best.weight_segments);
  }
  selected_source = best.instance_name;

  LOG(INFO) << "Found D2D source instance " << selected_source
            << " for model " << model_id
            << " with " << d2d_info.remote_addrs.size() << " remote addrs"
            << " and " << best.weight_segments.size() << " weight segments"
            << " (EPDT=" << best.epdt << ", candidates=" << candidates.size() << ")";

  // Unlock all instances except the selected source
  std::vector<std::string> instances_to_unlock;
  for (const auto& inst : locked_instances) {
    if (inst != selected_source) {
      instances_to_unlock.push_back(inst);
    }
  }
  model_mgr->release_d2d_locks(instances_to_unlock);

  return d2d_info;
}

// --- Dual-pool scheduling implementation ---

PoolType InstanceMgr::get_model_pool(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(allocation_mutex_);
  auto it = model_pool_assignments_.find(model_id);
  if (it != model_pool_assignments_.end()) {
    return it->second;
  }
  return PoolType::NONE;
}

bool InstanceMgr::is_steady_pool_instance(const std::string& instance_name) {
  for (const auto& bin : steady_bins_) {
    if (bin.instance_name == instance_name) return true;
  }
  return false;
}

int32_t InstanceMgr::steady_needed_gpus() {
  return static_cast<int32_t>(steady_bins_.size()) * options_.tensor_parallel_size()
      + pending_steady_gpus_;
}

int32_t InstanceMgr::compute_elastic_gpu_target_from_rate(double avg_token_rate) {
  if (avg_token_rate <= 0.0) return 1;
  // Exact capacity data: max token rate for 1..8 instances.
  static constexpr double kCapacity[] = {
      3000, 10000, 20000, 30000, 45000, 55000, 65000, 75000};
  static constexpr int32_t kCapacitySize =
      static_cast<int32_t>(sizeof(kCapacity) / sizeof(kCapacity[0]));
  for (int32_t i = 0; i < kCapacitySize; ++i) {
    if (kCapacity[i] > avg_token_rate) return i + 1;
  }
  // Beyond lookup table: extrapolate with linear regression.
  // Least-squares fit on the 8 data points: f(x) = 10702.38 * x - 10285.71
  static constexpr double kSlope = 10702.38;
  static constexpr double kNegIntercept = 10285.71;
  int32_t b = static_cast<int32_t>(
      std::ceil((avg_token_rate + kNegIntercept) / kSlope));
  return std::max(b, kCapacitySize + 1);
}

ResourceNeeds InstanceMgr::get_model_resource_needs(const std::string& model_id) {
  auto model_mgr = get_model_instance_mgr(model_id);
  auto stats = model_mgr ? model_mgr->get_traffic_stats()
                         : ModelInstanceMgr::TrafficStats{};

  const auto& gp_id = resolve_gp_model_id(model_id);
  auto it = model_resource_models_.find(gp_id);
  if (it != model_resource_models_.end()) {
    return it->second->calc_3d_resources(
        stats.token_rate, stats.avg_input_len,
        stats.avg_input_len2, stats.avg_output_len);
  }
  // Default: use hbm_b=20GB, compute_b=0, bandwidth=0 for unknown models
  return {20.0, 0.0, 0.0};
}

int32_t InstanceMgr::compute_gpu_target_for_model(const std::string& model_id) {
  auto model_mgr = get_model_instance_mgr(model_id);
  if (!model_mgr) return 1;

  auto stats = model_mgr->get_traffic_stats();
  if (stats.token_rate <= 0.0) return 1;

  const auto& gp_id = resolve_gp_model_id(model_id);
  auto it = model_resource_models_.find(gp_id);
  if (it == model_resource_models_.end()) return 1;

  ResourceNeeds needs = it->second->calc_3d_resources(
      stats.token_rate, stats.avg_input_len,
      stats.avg_input_len2, stats.avg_output_len);

  // HBM needs normalization; compute_sm and bandwidth are already [0,1] per GPU
  double hbm_gpus = needs.hbm_gb / gpu_hw_spec_.hbm_per_gpu_gb;
  double compute_gpus = needs.compute_sm;
  double bandwidth_gpus = needs.bandwidth;

  int32_t target = static_cast<int32_t>(
      std::ceil(std::max({hbm_gpus, compute_gpus, bandwidth_gpus})));
  return std::max(target, 1);
}

void InstanceMgr::assign_model_to_pool(const std::string& model_id) {
  std::unique_lock<std::mutex> alloc_lock(allocation_mutex_);

  // Already assigned?
  if (model_pool_assignments_.count(model_id) &&
      model_pool_assignments_[model_id] != PoolType::NONE) {
    return;
  }

  auto model_mgr = get_model_instance_mgr(model_id);
  if (!model_mgr) {
    LOG(ERROR) << "assign_model_to_pool: no model mgr for " << model_id;
    return;
  }

  // When steady pool is disabled, always assign to elastic pool
  if (options_.disable_steady_pool()) {
    model_pool_assignments_[model_id] = PoolType::ELASTIC;
    LOG(INFO) << "assign_model_to_pool: model=" << model_id
              << " -> ELASTIC (steady pool disabled)";
    return;
  }

  // When elastic pool is disabled, force steady pool assignment
  if (options_.disable_elastic_pool()) {
    ResourceNeeds needs = get_model_resource_needs(model_id);
    std::string instance = find_or_create_steady_bin(model_id, needs);
    if (instance.empty()) {
      LOG(WARNING) << "assign_model_to_pool: no instance for steady pool, "
                   << "model " << model_id << " unassigned (elastic disabled)";
      return;
    }

    model_pool_assignments_[model_id] = PoolType::STEADY;
    uint64_t model_size = get_model_size_bytes(model_id);
    model_mgr->set_model_state(instance, ModelState::ALLOCATED);
    deduct_free_pages(instance, model_size);
    {
      std::unique_lock<std::shared_mutex> lock(tag_mutex_);
      instance_tag_map_[instance] = InstanceTag::NORMAL;
      LOG(INFO) << "Tag change: instance " << instance
                << " -> NORMAL (steady assign, elastic pool disabled, model "
                << model_id << ")";
    }

    alloc_lock.unlock();
    send_model_wakeup(instance, model_id, true);

    LOG(INFO) << "assign_model_to_pool: model=" << model_id
              << " -> STEADY on " << instance << " (elastic pool disabled)";
    return;
  }

  // Use token rate threshold to decide pool: if rate exceeds single-instance
  // capacity (3000 tok/s), the model needs elastic pool with multiple instances.
  static constexpr double kSingleInstanceCapacity = 3000.0;
  auto stats = model_mgr->get_traffic_stats();
  bool needs_elastic = stats.token_rate > kSingleInstanceCapacity;

  LOG(INFO) << "assign_model_to_pool: model=" << model_id
            << " token_rate=" << stats.token_rate
            << " threshold=" << kSingleInstanceCapacity
            << " -> " << (needs_elastic ? "ELASTIC" : "STEADY");

  if (!needs_elastic) {
    // Steady pool
    ResourceNeeds needs = get_model_resource_needs(model_id);

    std::string instance = find_or_create_steady_bin(model_id, needs);
    if (instance.empty()) {
      LOG(WARNING) << "assign_model_to_pool: no instance for steady pool, "
                   << "falling back to elastic for " << model_id;
      model_pool_assignments_[model_id] = PoolType::ELASTIC;
      return;
    }

    model_pool_assignments_[model_id] = PoolType::STEADY;

    uint64_t model_size = get_model_size_bytes(model_id);
    model_mgr->set_model_state(instance, ModelState::ALLOCATED);
    deduct_free_pages(instance, model_size);
    {
      std::unique_lock<std::shared_mutex> lock(tag_mutex_);
      instance_tag_map_[instance] = InstanceTag::NORMAL;
      LOG(INFO) << "Tag change: instance " << instance
                << " -> NORMAL (steady pool assign for model " << model_id << ")";
    }

    alloc_lock.unlock();
    send_model_wakeup(instance, model_id, true);

    LOG(INFO) << "Assigned model " << model_id << " to STEADY pool on " << instance;
  } else {
    // Elastic pool — verify at least 2 instances are available.
    int32_t elastic_budget =
        std::max(0, total_available_gpus_.load() - steady_needed_gpus());
    int32_t elastic_used =
        static_cast<int32_t>(elastic_occupied_instances_.size());
    int32_t free_for_elastic = elastic_budget - elastic_used;

    if (free_for_elastic < 2) {
      LOG(INFO) << "assign_model_to_pool: elastic budget insufficient for "
                << model_id << " (free=" << free_for_elastic
                << " < 2), falling back to STEADY";

      ResourceNeeds needs = get_model_resource_needs(model_id);

      std::string instance = find_or_create_steady_bin(model_id, needs);
      if (instance.empty()) {
        LOG(WARNING) << "assign_model_to_pool: no instance for steady pool "
                     << "fallback either, model " << model_id << " unassigned";
        return;
      }

      model_pool_assignments_[model_id] = PoolType::STEADY;
      uint64_t model_size = get_model_size_bytes(model_id);
      model_mgr->set_model_state(instance, ModelState::ALLOCATED);
      deduct_free_pages(instance, model_size);
      {
        std::unique_lock<std::shared_mutex> lock(tag_mutex_);
        instance_tag_map_[instance] = InstanceTag::NORMAL;
        LOG(INFO) << "Tag change: instance " << instance
                  << " -> NORMAL (elastic fallback to steady for model "
                  << model_id << ")";
      }

      alloc_lock.unlock();
      send_model_wakeup(instance, model_id, true);

      LOG(INFO) << "Assigned model " << model_id
                << " to STEADY pool (elastic budget insufficient) on "
                << instance;
    } else {
      model_pool_assignments_[model_id] = PoolType::ELASTIC;
      LOG(INFO) << "Assigned model " << model_id
                << " to ELASTIC pool";
    }
  }
}

std::string InstanceMgr::find_or_create_steady_bin(
    const std::string& model_id, const ResourceNeeds& needs,
    bool allow_reclaim) {

  // Phase 1: min cos heuristic — select the feasible bin whose load vector
  // is most orthogonal to the item's resource vector (minimizes cosine).
  {
    const int32_t max_models = FLAGS_max_models_per_gpu_in_steady_pool;
    SteadyBin* best_bin = nullptr;
    double best_cos = std::numeric_limits<double>::max();

    for (auto& bin : steady_bins_) {
      if (max_models > 0 &&
          static_cast<int32_t>(bin.models.size()) >= max_models) {
        continue;
      }
      if (bin.remaining_hbm_gb >= needs.hbm_gb &&
          bin.remaining_compute_sm >= needs.compute_sm &&
          bin.remaining_bandwidth >= needs.bandwidth) {
        double cos_val = compute_cos_similarity(needs, bin, gpu_hw_spec_);
        if (cos_val < best_cos) {
          best_cos = cos_val;
          best_bin = &bin;
        }
      }
    }
    if (best_bin) {
      best_bin->remaining_hbm_gb -= needs.hbm_gb;
      best_bin->remaining_compute_sm -= needs.compute_sm;
      best_bin->remaining_bandwidth -= needs.bandwidth;
      best_bin->models.insert(model_id);
      LOG(INFO) << "Placed model " << model_id << " in steady bin "
                << best_bin->instance_name << " (min_cos=" << best_cos
                << ", remaining hbm=" << best_bin->remaining_hbm_gb
                << "GB, compute=" << best_bin->remaining_compute_sm
                << ", bandwidth=" << best_bin->remaining_bandwidth << ")";
      return best_bin->instance_name;
    }
  }

  // Phase 2: Open new bin from idle instances
  {
    std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
    for (const auto& [inst_name, _] : instances_) {
      if (get_instance_tag(inst_name) != InstanceTag::NONE) continue;
      if (!has_valid_xtensor_info(inst_name)) continue;

      bool has_existing_bin = false;
      for (const auto& bin : steady_bins_) {
        if (bin.instance_name == inst_name) { has_existing_bin = true; break; }
      }
      if (has_existing_bin) continue;

      if (elastic_occupied_instances_.count(inst_name)) continue;

      uint64_t free_bytes = get_instance_free_bytes(inst_name);
      double free_gb = static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0);
      if (free_gb < needs.hbm_gb) {
        LOG(INFO) << "find_or_create_steady_bin: skip " << inst_name
                  << " for model " << model_id << " (free=" << free_gb
                  << "GB < need=" << needs.hbm_gb << "GB)";
        continue;
      }

      SteadyBin new_bin;
      new_bin.instance_name = inst_name;
      new_bin.remaining_hbm_gb = free_gb - needs.hbm_gb;
      new_bin.remaining_compute_sm = gpu_hw_spec_.compute_sm_per_gpu - needs.compute_sm;
      new_bin.remaining_bandwidth = gpu_hw_spec_.bandwidth_per_gpu - needs.bandwidth;
      new_bin.models.insert(model_id);
      steady_bins_.push_back(std::move(new_bin));

      LOG(INFO) << "Created new steady bin on " << inst_name << " for model "
                << model_id;
      return inst_name;
    }
  }

  // Phase 3: No idle instance — reclaim from elastic pool (wait with timeout)
  if (!allow_reclaim) {
    return "";
  }
  pending_steady_gpus_ += options_.tensor_parallel_size();

  // Trigger auto-scaling thread to run immediately (will shrink elastic pool).
  {
    std::lock_guard<std::mutex> lock(scaling_trigger_mutex_);
    scaling_requested_ = true;
  }
  scaling_trigger_cv_.notify_one();

  // Wait for a NONE-tagged instance to appear (max 10s).
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  std::string found_instance;

  while (found_instance.empty() && std::chrono::steady_clock::now() < deadline) {
    // Release allocation_mutex_ to let drain threads finish.
    allocation_mutex_.unlock();
    {
      std::unique_lock<std::mutex> wait_lock(instance_freed_mutex_);
      instance_freed_cv_.wait_for(wait_lock, std::chrono::seconds(1));
    }
    allocation_mutex_.lock();

    // Scan for freed instance.
    std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
    for (const auto& [inst_name, _] : instances_) {
      if (get_instance_tag(inst_name) != InstanceTag::NONE) continue;
      if (!has_valid_xtensor_info(inst_name)) continue;

      bool has_existing_bin = false;
      for (const auto& bin : steady_bins_) {
        if (bin.instance_name == inst_name) { has_existing_bin = true; break; }
      }
      if (has_existing_bin) continue;

      if (elastic_occupied_instances_.count(inst_name)) continue;

      uint64_t free_bytes = get_instance_free_bytes(inst_name);
      double free_gb = static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0);
      if (free_gb < needs.hbm_gb) {
        LOG(INFO) << "find_or_create_steady_bin (reclaim): skip " << inst_name
                  << " for model " << model_id << " (free=" << free_gb
                  << "GB < need=" << needs.hbm_gb << "GB)";
        continue;
      }

      SteadyBin new_bin;
      new_bin.instance_name = inst_name;
      new_bin.remaining_hbm_gb = free_gb - needs.hbm_gb;
      new_bin.remaining_compute_sm = gpu_hw_spec_.compute_sm_per_gpu - needs.compute_sm;
      new_bin.remaining_bandwidth = gpu_hw_spec_.bandwidth_per_gpu - needs.bandwidth;
      new_bin.models.insert(model_id);
      steady_bins_.push_back(std::move(new_bin));
      found_instance = inst_name;
      break;
    }
  }

  pending_steady_gpus_ -= options_.tensor_parallel_size();

  if (!found_instance.empty()) {
    LOG(INFO) << "Reclaimed instance " << found_instance
              << " for steady bin, model " << model_id;
  }
  return found_instance;
}

std::vector<std::tuple<std::string, std::string, std::string>>
InstanceMgr::remove_model_from_steady_bin(const std::string& model_id) {
  // Must be called with allocation_mutex_ held
  std::vector<std::tuple<std::string, std::string, std::string>> moves;

  for (auto it = steady_bins_.begin(); it != steady_bins_.end(); ++it) {
    if (it->models.count(model_id)) {
      // Reclaim resources
      ResourceNeeds needs = get_model_resource_needs(model_id);
      it->remaining_hbm_gb += needs.hbm_gb;
      it->remaining_compute_sm += needs.compute_sm;
      it->remaining_bandwidth += needs.bandwidth;
      it->models.erase(model_id);

      LOG(INFO) << "Removed model " << model_id << " from steady bin "
                << it->instance_name;

      // If bin is now empty, remove it
      if (it->models.empty()) {
        LOG(INFO) << "Steady bin " << it->instance_name << " is now empty, releasing";
        steady_bins_.erase(it);
        return moves;
      }

      // Check imbalance ratio R(B) = min(S_k) / max(S_k)
      // S_k = normalized load = (capacity - remaining) / capacity
      double s1 = (gpu_hw_spec_.hbm_per_gpu_gb - it->remaining_hbm_gb)
                   / gpu_hw_spec_.hbm_per_gpu_gb;
      double s2 = (gpu_hw_spec_.compute_sm_per_gpu - it->remaining_compute_sm)
                   / gpu_hw_spec_.compute_sm_per_gpu;
      double s3 = (gpu_hw_spec_.bandwidth_per_gpu - it->remaining_bandwidth)
                   / gpu_hw_spec_.bandwidth_per_gpu;

      double s_min = std::min({s1, s2, s3});
      double s_max = std::max({s1, s2, s3});
      double R = (s_max > 1e-12) ? (s_min / s_max) : 1.0;

      if (R >= 0.5) {
        return moves;  // balanced enough
      }

      // R(B) < 0.5 — trigger per-bin repack (bin-packing §2.3)
      LOG(INFO) << "Bin " << it->instance_name << " imbalance ratio R=" << R
                << " < 0.5, triggering per-bin repack";

      std::string old_instance = it->instance_name;
      std::vector<std::string> displaced_models(it->models.begin(),
                                                it->models.end());

      // Erase the unbalanced bin
      steady_bins_.erase(it);

      // Re-insert each displaced model via min cos (no elastic reclamation)
      std::vector<std::string> fallback_models;
      for (const auto& mid : displaced_models) {
        ResourceNeeds mid_needs = get_model_resource_needs(mid);
        std::string new_instance =
            find_or_create_steady_bin(mid, mid_needs, /*allow_reclaim=*/false);

        if (new_instance.empty()) {
          // Couldn't place — will go to fallback bin on old_instance
          fallback_models.push_back(mid);
        } else if (new_instance != old_instance) {
          moves.emplace_back(mid, old_instance, new_instance);
        }
        // if new_instance == old_instance, model stays put (no move needed)
      }

      // Create fallback bin on old_instance for unplaced models
      if (!fallback_models.empty()) {
        SteadyBin fallback_bin;
        fallback_bin.instance_name = old_instance;
        fallback_bin.remaining_hbm_gb = gpu_hw_spec_.hbm_per_gpu_gb;
        fallback_bin.remaining_compute_sm = gpu_hw_spec_.compute_sm_per_gpu;
        fallback_bin.remaining_bandwidth = gpu_hw_spec_.bandwidth_per_gpu;
        for (const auto& mid : fallback_models) {
          ResourceNeeds mid_needs = get_model_resource_needs(mid);
          fallback_bin.remaining_hbm_gb -= mid_needs.hbm_gb;
          fallback_bin.remaining_compute_sm -= mid_needs.compute_sm;
          fallback_bin.remaining_bandwidth -= mid_needs.bandwidth;
          fallback_bin.models.insert(mid);
        }
        steady_bins_.push_back(std::move(fallback_bin));
        LOG(INFO) << "Created fallback bin on " << old_instance << " for "
                  << fallback_models.size() << " unplaced models";
      }

      return moves;
    }
  }
  return moves;
}

bool InstanceMgr::steady_part_check_upgrading(const std::string& model_id) {
  if (options_.disable_steady_pool()) {
    return false;
  }
  if (options_.disable_elastic_pool()) {
    return false;
  }

  std::unique_lock<std::mutex> alloc_lock(allocation_mutex_);

  auto pool_it = model_pool_assignments_.find(model_id);
  if (pool_it == model_pool_assignments_.end() ||
      pool_it->second != PoolType::STEADY) {
    return false;
  }

  auto model_mgr = get_model_instance_mgr(model_id);
  if (!model_mgr) return false;

  // Use token rate threshold (same as assign_model_to_pool)
  static constexpr double kSingleInstanceCapacity = 3000.0;
  auto stats = model_mgr->get_traffic_stats();
  if (stats.token_rate <= kSingleInstanceCapacity) {
    return false;  // still fits in steady pool
  }

  // Don't upgrade if elastic pool can't guarantee at least 1P1D per model.
  int32_t steady_gpus = steady_needed_gpus();
  int32_t elastic_model_count = 0;
  for (const auto& [mid, pool] : model_pool_assignments_) {
    if (pool == PoolType::ELASTIC) {
      ++elastic_model_count;
    }
  }
  if (steady_gpus + (elastic_model_count + 1) * 2 >
      total_available_gpus_.load()) {
    return false;
  }

  // Find the steady instance hosting this model
  std::string steady_instance;
  for (const auto& bin : steady_bins_) {
    if (bin.models.count(model_id)) {
      steady_instance = bin.instance_name;
      break;
    }
  }

  // Don't upgrade while the model wakeup is still in progress (ALLOCATED state).
  // The drain thread would fail (ALLOCATED→DRAINING is invalid), and
  // dynamic_part_auto_scaling would see the in-progress allocation as already
  // satisfying the budget, resulting in no new instances being woken.
  if (!steady_instance.empty() &&
      model_mgr->get_model_state(steady_instance) != ModelState::WAKEUP) {
    return false;
  }

  // Upgrade to elastic
  pool_it->second = PoolType::ELASTIC;

  auto repack_moves = remove_model_from_steady_bin(model_id);

  alloc_lock.unlock();

  LOG(INFO) << "Upgrading model " << model_id << " from STEADY to ELASTIC pool ";

  // Async: drain + sleep the model on old steady instance
  if (!steady_instance.empty()) {
    std::string inst = steady_instance;
    std::string mid = model_id;
    std::thread([this, inst, mid]() {
      auto mgr = get_model_instance_mgr(mid);
      if (mgr) mgr->set_model_state(inst, ModelState::DRAINING);
      if (!wait_for_model_drain(inst, mid)) return;
      send_model_sleep(inst, mid);
    }).detach();
  }

  // Execute R(B) repack moves: wake on new instance, drain+sleep on old
  for (const auto& [mid, old_inst, new_inst] : repack_moves) {
    LOG(INFO) << "R(B) repack: moving model " << mid
              << " from " << old_inst << " to " << new_inst;
    auto mgr = get_model_instance_mgr(mid);
    if (mgr) {
      mgr->set_model_state(new_inst, ModelState::ALLOCATED);
    }
    deduct_free_pages(new_inst, get_model_size_bytes(mid));
    {
      std::unique_lock<std::shared_mutex> lock(tag_mutex_);
      instance_tag_map_[new_inst] = InstanceTag::NORMAL;
      LOG(INFO) << "Tag change: instance " << new_inst
                << " -> NORMAL (R(B) repack for model " << mid << ")";
    }
    bool wakeup_ok = send_model_wakeup(new_inst, mid, false);
    if (!wakeup_ok) {
      LOG(WARNING) << "R(B) repack: wakeup failed for " << mid
                   << " on " << new_inst
                   << ", rolling back and keeping old instance " << old_inst;
      // Roll back new instance state
      if (mgr) {
        mgr->set_model_state(new_inst, ModelState::SLEEP);
      }
      if (count_active_models_on_instance(new_inst) == 0) {
        std::unique_lock<std::shared_mutex> lock(tag_mutex_);
        instance_tag_map_[new_inst] = InstanceTag::NONE;
      }
      continue;
    }
    if (mgr) mgr->set_model_state(old_inst, ModelState::DRAINING);
    wait_for_model_drain(old_inst, mid);
    send_model_sleep(old_inst, mid);
  }

  return true;
}

bool InstanceMgr::route_to_steady_instance(std::shared_ptr<Request> request) {
  std::lock_guard<std::mutex> lock(allocation_mutex_);
  for (const auto& bin : steady_bins_) {
    if (bin.models.count(request->model)) {
      request->routing.prefill_name = bin.instance_name;
      request->routing.decode_name = bin.instance_name;
      return true;
    }
  }
  LOG(ERROR) << "route_to_steady_instance: model " << request->model
             << " not found in any steady bin";
  return false;
}

std::vector<std::tuple<std::string, std::string, std::string>>
InstanceMgr::repack_steady_bins() {
  // Must be called with allocation_mutex_ held
  struct PackItem {
    std::string model_id;
    ResourceNeeds needs;
    std::string old_instance;
    double dominant_ratio;
  };

  // Collect all items from current bins
  std::vector<PackItem> items;
  for (const auto& bin : steady_bins_) {
    for (const auto& model_id : bin.models) {
      PackItem item;
      item.model_id = model_id;
      item.needs = get_model_resource_needs(model_id);
      item.old_instance = bin.instance_name;
      item.dominant_ratio = std::max({
          item.needs.hbm_gb / gpu_hw_spec_.hbm_per_gpu_gb,
          item.needs.compute_sm / gpu_hw_spec_.compute_sm_per_gpu,
          item.needs.bandwidth / gpu_hw_spec_.bandwidth_per_gpu});
      items.push_back(std::move(item));
    }
  }

  if (items.empty()) {
    steady_bins_.clear();
    return {};
  }

  // Sort by dominant resource ratio (decreasing) — FFD heuristic
  std::sort(items.begin(), items.end(),
            [](const PackItem& a, const PackItem& b) {
              return a.dominant_ratio > b.dominant_ratio;
            });

  // Collect old bin instance names for reuse
  std::vector<std::string> old_instances;
  for (const auto& bin : steady_bins_) {
    old_instances.push_back(bin.instance_name);
  }

  // Build new bins with FFD + min cos placement
  std::vector<SteadyBin> new_bins;
  size_t next_old_instance = 0;
  const int32_t max_models = FLAGS_max_models_per_gpu_in_steady_pool;

  for (const auto& item : items) {
    bool placed = false;

    // min cos: find the feasible bin with minimum cosine similarity
    SteadyBin* best = nullptr;
    double best_cos = std::numeric_limits<double>::max();
    for (auto& bin : new_bins) {
      if (max_models > 0 &&
          static_cast<int32_t>(bin.models.size()) >= max_models) {
        continue;
      }
      if (bin.remaining_hbm_gb >= item.needs.hbm_gb &&
          bin.remaining_compute_sm >= item.needs.compute_sm &&
          bin.remaining_bandwidth >= item.needs.bandwidth) {
        double cos_val = compute_cos_similarity(item.needs, bin, gpu_hw_spec_);
        if (cos_val < best_cos) {
          best_cos = cos_val;
          best = &bin;
        }
      }
    }
    if (best) {
      best->remaining_hbm_gb -= item.needs.hbm_gb;
      best->remaining_compute_sm -= item.needs.compute_sm;
      best->remaining_bandwidth -= item.needs.bandwidth;
      best->models.insert(item.model_id);
      placed = true;
    }
    if (!placed) {
      // Open new bin, preferring old steady instances
      std::string inst_name;
      if (next_old_instance < old_instances.size()) {
        inst_name = old_instances[next_old_instance++];
      } else {
        // Need a completely new instance — find idle
        std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
        for (const auto& [name, _] : instances_) {
          if (elastic_occupied_instances_.count(name)) continue;
          if (!has_valid_xtensor_info(name)) continue;
          uint64_t free_bytes = get_instance_free_bytes(name);
          double free_gb = static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0);
          if (free_gb < item.needs.hbm_gb) {
            LOG(INFO) << "repack_steady_bins: skip " << name
                      << " for model " << item.model_id << " (free=" << free_gb
                      << "GB < need=" << item.needs.hbm_gb << "GB)";
            continue;
          }
          // Check not already used in new_bins
          bool already_used = false;
          for (const auto& nb : new_bins) {
            if (nb.instance_name == name) { already_used = true; break; }
          }
          if (already_used) continue;
          inst_name = name;
          break;
        }
      }

      if (inst_name.empty()) {
        LOG(ERROR) << "repack_steady_bins: no instance available for model "
                   << item.model_id;
        continue;
      }

      SteadyBin new_bin;
      new_bin.instance_name = inst_name;
      new_bin.remaining_hbm_gb = gpu_hw_spec_.hbm_per_gpu_gb - item.needs.hbm_gb;
      new_bin.remaining_compute_sm = gpu_hw_spec_.compute_sm_per_gpu - item.needs.compute_sm;
      new_bin.remaining_bandwidth = gpu_hw_spec_.bandwidth_per_gpu - item.needs.bandwidth;
      new_bin.models.insert(item.model_id);
      new_bins.push_back(std::move(new_bin));
    }
  }

  // Compute moves
  std::vector<std::tuple<std::string, std::string, std::string>> moves;
  for (const auto& item : items) {
    std::string new_instance;
    for (const auto& bin : new_bins) {
      if (bin.models.count(item.model_id)) {
        new_instance = bin.instance_name;
        break;
      }
    }
    if (!new_instance.empty() && new_instance != item.old_instance) {
      moves.emplace_back(item.model_id, item.old_instance, new_instance);
    }
  }

  steady_bins_ = std::move(new_bins);

  LOG(INFO) << "Steady pool repack: " << items.size() << " models, "
            << steady_bins_.size() << " bins, " << moves.size() << " moves";

  return moves;
}

void InstanceMgr::steady_part_auto_repacking() {
  if (options_.disable_steady_pool()) {
    return;
  }

  std::unique_lock<std::mutex> alloc_lock(allocation_mutex_);

  // Phase 1: collect models with 0 heat → schedule sleep
  std::vector<std::pair<std::string, std::string>> models_to_remove;  // (model_id, instance)
  for (const auto& bin : steady_bins_) {
    for (const auto& model_id : bin.models) {
      auto model_mgr = get_model_instance_mgr(model_id);
      if (model_mgr && model_mgr->get_model_heat() == 0) {
        models_to_remove.emplace_back(model_id, bin.instance_name);
      }
    }
  }

  for (const auto& [model_id, instance] : models_to_remove) {
    remove_model_from_steady_bin(model_id);
    model_pool_assignments_.erase(model_id);
    // Query actual awake instances (bin instance may be stale after cascading repacks)
    auto model_mgr = get_model_instance_mgr(model_id);
    if (model_mgr) {
      auto awake_insts = model_mgr->get_awake_instances();
      for (const auto& awake_inst : awake_insts) {
        LOG(INFO) << "steady_part_auto_repacking: sleeping model " << model_id
                  << " on " << awake_inst << " (heat=0)";
        std::string inst_copy = awake_inst;
        std::string mid = model_id;
        std::thread([this, inst_copy, mid]() {
          send_model_sleep(inst_copy, mid);
        }).detach();
      }
    }
  }

  // Phase 2: FFD repack of remaining models
  auto moves = repack_steady_bins();

  alloc_lock.unlock();

  // Phase 3: Execute moves (wake on new, drain+sleep on old)
  for (const auto& [model_id, old_inst, new_inst] : moves) {
    LOG(INFO) << "steady_part_auto_repacking: moving model " << model_id
              << " from " << old_inst << " to " << new_inst;
    auto mgr = get_model_instance_mgr(model_id);
    if (mgr) {
      mgr->set_model_state(new_inst, ModelState::ALLOCATED);
    }
    deduct_free_pages(new_inst, get_model_size_bytes(model_id));
    {
      std::unique_lock<std::shared_mutex> lock(tag_mutex_);
      instance_tag_map_[new_inst] = InstanceTag::NORMAL;
      LOG(INFO) << "Tag change: instance " << new_inst
                << " -> NORMAL (steady repack for model " << model_id << ")";
    }
    bool wakeup_ok = send_model_wakeup(new_inst, model_id, false);
    if (!wakeup_ok) {
      LOG(WARNING) << "steady_part_auto_repacking: wakeup failed for "
                   << model_id << " on " << new_inst
                   << ", keeping old instance " << old_inst;
      continue;
    }
    if (mgr) mgr->set_model_state(old_inst, ModelState::DRAINING);
    wait_for_model_drain(old_inst, model_id);
    send_model_sleep(old_inst, model_id);
  }
}

void InstanceMgr::elastic_to_steady_demotion() {
  if (options_.disable_steady_pool()) {
    return;
  }
  if (options_.disable_elastic_pool()) {
    return;
  }

  static constexpr int kDemotionThresholdSeconds = 30;
  auto now = std::chrono::steady_clock::now();

  // Phase 1: Scan elastic models and update low-demand tracking
  struct DemotionCandidate {
    std::string model_id;
    std::shared_ptr<ModelInstanceMgr> mgr;
  };
  std::vector<DemotionCandidate> candidates;

  {
    std::unique_lock<std::mutex> alloc_lock(allocation_mutex_);

    std::vector<std::string> elastic_models;
    for (const auto& [id, pool] : model_pool_assignments_) {
      if (pool == PoolType::ELASTIC) {
        elastic_models.push_back(id);
      }
    }

    for (const auto& model_id : elastic_models) {
      auto model_mgr = get_model_instance_mgr(model_id);
      if (!model_mgr) continue;

      auto stats = model_mgr->get_traffic_stats();
      int32_t gpu_target = compute_elastic_gpu_target_from_rate(stats.token_rate);
      if (model_mgr->get_model_heat() == 0) gpu_target = 0;

      if (gpu_target != 1) {
        // Demand is not "low but nonzero" — reset tracking
        elastic_low_demand_since_.erase(model_id);
        continue;
      }

      // gpu_target == 1: track low demand duration
      auto it = elastic_low_demand_since_.find(model_id);
      if (it == elastic_low_demand_since_.end()) {
        elastic_low_demand_since_[model_id] = now;
        continue;
      }

      auto duration_s = std::chrono::duration_cast<std::chrono::seconds>(
          now - it->second).count();
      if (duration_s < kDemotionThresholdSeconds) {
        continue;
      }

      // Sustained low demand — mark as demotion candidate
      candidates.push_back({model_id, model_mgr});
    }
  }  // release allocation_mutex_

  // Phase 2: Attempt demotion for each candidate
  for (const auto& candidate : candidates) {
    const auto& model_id = candidate.model_id;
    auto model_mgr = candidate.mgr;

    std::unique_lock<std::mutex> alloc_lock(allocation_mutex_);

    // Re-verify: still ELASTIC and gpu_target still <= 1
    auto pool_it = model_pool_assignments_.find(model_id);
    if (pool_it == model_pool_assignments_.end() ||
        pool_it->second != PoolType::ELASTIC) {
      elastic_low_demand_since_.erase(model_id);
      continue;
    }

    auto stats = model_mgr->get_traffic_stats();
    int32_t gpu_target = compute_elastic_gpu_target_from_rate(stats.token_rate);
    if (model_mgr->get_model_heat() == 0) gpu_target = 0;
    if (gpu_target != 1) {
      elastic_low_demand_since_.erase(model_id);
      continue;
    }

    // Try to find a steady bin (no reclamation allowed)
    ResourceNeeds needs = get_model_resource_needs(model_id);
    std::string steady_instance =
        find_or_create_steady_bin(model_id, needs, /*allow_reclaim=*/false);

    if (steady_instance.empty()) {
      LOG(INFO) << "elastic_to_steady_demotion: no room in steady pool for "
                << model_id << ", skipping this round";
      continue;
    }

    // Collect elastic instances before changing pool
    auto elastic_instances = model_mgr->get_awake_instances();

    // Set model state to ALLOCATED on steady instance, deduct memory
    uint64_t model_size = get_model_size_bytes(model_id);
    model_mgr->set_model_state(steady_instance, ModelState::ALLOCATED);
    deduct_free_pages(steady_instance, model_size);
    {
      std::unique_lock<std::shared_mutex> lock(tag_mutex_);
      instance_tag_map_[steady_instance] = InstanceTag::NORMAL;
      LOG(INFO) << "Tag change: instance " << steady_instance
                << " -> NORMAL (elastic-to-steady demotion for model "
                << model_id << ")";
    }

    // Change pool assignment to STEADY
    pool_it->second = PoolType::STEADY;
    {
      std::unique_lock<std::shared_mutex> lock(tag_mutex_);
      clear_elastic_decode_slot_locked(
          model_id, "model demoted from ELASTIC to STEADY");
    }

    LOG(INFO) << "elastic_to_steady_demotion: demoting model " << model_id
              << " from ELASTIC to STEADY on " << steady_instance
              << " (elastic instances: " << elastic_instances.size() << ")";

    alloc_lock.unlock();

    // Blocking wakeup on steady instance
    send_model_wakeup(steady_instance, model_id, true);

    // Check if wakeup succeeded
    if (model_mgr->get_model_state(steady_instance) != ModelState::WAKEUP) {
      LOG(ERROR) << "elastic_to_steady_demotion: wakeup failed for "
                 << model_id << " on " << steady_instance << ", rolling back";
      std::lock_guard<std::mutex> lock(allocation_mutex_);
      remove_model_from_steady_bin(model_id);
      model_pool_assignments_[model_id] = PoolType::ELASTIC;
      elastic_low_demand_since_.erase(model_id);
      continue;
    }

    // Drain and sleep all elastic instances for this model
    for (const auto& elastic_inst : elastic_instances) {
      if (elastic_inst == steady_instance) continue;
      model_mgr->set_model_state(elastic_inst, ModelState::DRAINING);
      wait_for_model_drain(elastic_inst, model_id);
      send_model_sleep(elastic_inst, model_id);
      {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        elastic_occupied_instances_.erase(elastic_inst);
      }
    }

    {
      std::lock_guard<std::mutex> lock(allocation_mutex_);
      elastic_low_demand_since_.erase(model_id);
    }

    LOG(INFO) << "elastic_to_steady_demotion: model " << model_id
              << " successfully demoted to STEADY on " << steady_instance;
  }
}

std::vector<std::string> InstanceMgr::get_awake_decode_instances(const std::string& model_id) {
  auto model_mgr = get_model_instance_mgr(model_id);
  auto all_instances = model_mgr->get_awake_instances();

  std::vector<std::string> decode_instances;
  for (const auto& inst : all_instances) {
    auto tag = get_instance_tag(inst);
    if (tag == InstanceTag::DECODE) {
      decode_instances.push_back(inst);
    }
  }
  return decode_instances;
}

int InstanceMgr::count_awake_models_on_instance(const std::string& instance_name) {
  int count = 0;
  std::shared_lock<std::shared_mutex> lock(model_instance_mgr_mutex_);
  for (const auto& [model_id, mgr] : model_instance_mgrs_) {
    auto awake = mgr->get_awake_instances();
    for (const auto& inst : awake) {
      if (inst == instance_name) {
        ++count;
        break;
      }
    }
  }
  return count;
}

int InstanceMgr::count_active_models_on_instance(const std::string& instance_name) {
  int count = 0;
  std::shared_lock<std::shared_mutex> lock(model_instance_mgr_mutex_);
  for (const auto& [model_id, mgr] : model_instance_mgrs_) {
    auto active = mgr->get_active_instances();
    for (const auto& inst : active) {
      if (inst == instance_name) {
        ++count;
        break;
      }
    }
  }
  return count;
}

InstanceTag InstanceMgr::get_instance_tag(const std::string& instance_name) const {
  std::shared_lock<std::shared_mutex> lock(tag_mutex_);
  auto it = instance_tag_map_.find(instance_name);
  if (it != instance_tag_map_.end()) {
    return it->second;
  }
  return InstanceTag::NONE;
}

void InstanceMgr::set_instance_tag(const std::string& instance_name,
                                   InstanceTag tag) {
  std::unique_lock<std::shared_mutex> lock(tag_mutex_);
  instance_tag_map_[instance_name] = tag;
  LOG(INFO) << "Tag change: Set tag for " << instance_name << " to "
            << instance_tag_name(tag);
}

InstanceTag InstanceMgr::determine_elastic_tag(const std::string& model_id) {
  std::unique_lock<std::shared_mutex> lock(tag_mutex_);
  return determine_elastic_tag_locked(model_id);
}

InstanceTag InstanceMgr::determine_elastic_tag_locked(const std::string& model_id) {
  // Self-heal stale reservation before making a new decision.
  refresh_elastic_decode_slot_locked(model_id);
  const bool reserved_before =
      (elastic_decode_reserved_models_.count(model_id) > 0);
  const bool active_decode_slot = has_elastic_decode_slot_locked(model_id);
  InstanceTag decided_tag = InstanceTag::DECODE;
  if (reserved_before || active_decode_slot) {
    elastic_decode_reserved_models_.insert(model_id);
    decided_tag = InstanceTag::PREFILL;
  } else {
    elastic_decode_reserved_models_.insert(model_id);
    decided_tag = InstanceTag::DECODE;
  }
  LOG(INFO) << "Elastic tag decision: model=" << model_id
            << " reserved_before=" << reserved_before
            << " active_decode_slot=" << active_decode_slot
            << " decided=" << instance_tag_name(decided_tag);
  return decided_tag;
}

bool InstanceMgr::has_elastic_decode_slot_locked(const std::string& model_id) {
  auto model_mgr = get_model_instance_mgr(model_id);
  if (!model_mgr) return false;

  auto all_instances = model_mgr->get_all_instance_names();
  for (const auto& inst : all_instances) {
    auto state = model_mgr->get_model_state(inst);
    if (state != ModelState::WAKEUP && state != ModelState::ALLOCATED) {
      continue;
    }
    auto tag_it = instance_tag_map_.find(inst);
    if (tag_it != instance_tag_map_.end() &&
        tag_it->second == InstanceTag::DECODE) {
      return true;
    }
  }
  return false;
}

void InstanceMgr::refresh_elastic_decode_slot_locked(const std::string& model_id) {
  if (elastic_decode_reserved_models_.count(model_id) == 0) {
    return;
  }
  if (has_elastic_decode_slot_locked(model_id)) {
    return;
  }
  clear_elastic_decode_slot_locked(
      model_id, "no DECODE instance in WAKEUP/ALLOCATED");
}

void InstanceMgr::clear_elastic_decode_slot_locked(const std::string& model_id,
                                                   const std::string& reason) {
  if (elastic_decode_reserved_models_.erase(model_id) > 0) {
    LOG(INFO) << "Elastic decode reservation reset for model " << model_id
              << " (" << reason << ")";
  }
}

void InstanceMgr::promote_prefill_to_decode(const std::string& model_id) {
  {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    auto pool_it = model_pool_assignments_.find(model_id);
    if (pool_it == model_pool_assignments_.end() ||
        pool_it->second != PoolType::ELASTIC) {
      return;
    }
  }

  auto model_mgr = get_model_instance_mgr(model_id);
  if (!model_mgr) return;

  // Hold tag_mutex_ across check-and-promote to prevent two concurrent
  // drain threads from both promoting a PREFILL → DECODE.
  std::unique_lock<std::shared_mutex> tag_lock(tag_mutex_);

  // Check if a DECODE instance already exists (under lock).
  auto awake = model_mgr->get_awake_instances();
  for (const auto& inst : awake) {
    auto it = instance_tag_map_.find(inst);
    if (it != instance_tag_map_.end() && it->second == InstanceTag::DECODE) {
      return;  // Already has DECODE, nothing to do.
    }
  }

  // Promote first PREFILL to DECODE.
  for (const auto& inst : awake) {
    auto it = instance_tag_map_.find(inst);
    if (it != instance_tag_map_.end() && it->second == InstanceTag::PREFILL) {
      it->second = InstanceTag::DECODE;
      elastic_decode_reserved_models_.insert(model_id);
      LOG(INFO) << "Tag change: Promoted instance " << inst << " from PREFILL to DECODE "
                << "for model " << model_id;
      return;
    }
  }
}

void InstanceMgr::link_instance_bidirectional(
    const std::string& instance_name,
    const std::vector<std::string>& peer_names) {
  auto parse_device_addr = [](const std::string& device_addr,
                              std::string* device_ip,
                              uint32_t* port) -> bool {
    const auto pos = device_addr.rfind(':');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= device_addr.size()) {
      return false;
    }
    std::string ip = device_addr.substr(0, pos);
    std::string port_str = device_addr.substr(pos + 1);
    try {
      unsigned long parsed = std::stoul(port_str);
      if (parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
      }
      *device_ip = std::move(ip);
      *port = static_cast<uint32_t>(parsed);
      return true;
    } catch (...) {
      return false;
    }
  };

  auto build_link_instance_req =
      [&parse_device_addr](const std::string& source_name,
                           const InstanceMetaInfo& source_info,
                           janus::proto::InstanceClusterInfo* req) -> bool {
    if (source_name.empty() || req == nullptr) {
      return false;
    }
    if (source_info.cluster_ids.empty() || source_info.addrs.empty() ||
        source_info.device_addrs.empty() || source_info.dp_size <= 0) {
      return false;
    }
    if (source_info.cluster_ids.size() != source_info.addrs.size() ||
        source_info.cluster_ids.size() != source_info.device_addrs.size()) {
      return false;
    }

    req->set_instance_name(source_name);
    for (const auto cluster_id : source_info.cluster_ids) {
      req->add_cluster_ids(cluster_id);
    }
    for (const auto& addr : source_info.addrs) {
      req->add_addrs(addr);
    }
    for (const auto& device_addr : source_info.device_addrs) {
      std::string ip;
      uint32_t port = 0;
      if (!parse_device_addr(device_addr, &ip, &port)) {
        return false;
      }
      req->add_device_ips(ip);
      req->add_ports(port);
    }
    req->set_dp_size(source_info.dp_size);
    return true;
  };

  auto do_link_instance =
      [this](const std::string& target_name,
             const std::string& target_rpc_address,
             const janus::proto::InstanceClusterInfo& req) -> bool {
    brpc::Channel rpc_channel;
    brpc::ChannelOptions rpc_options;
    rpc_options.timeout_ms = options_.timeout_ms();
    rpc_options.max_retry = 3;
    std::string load_balancer = "";
    if (rpc_channel.Init(
            target_rpc_address.c_str(), load_balancer.c_str(), &rpc_options) !=
        0) {
      LOG(WARNING) << "Failed to init LinkInstance channel to " << target_name
                   << " (" << target_rpc_address << ")";
      return false;
    }

    janus::proto::DisaggPDService_Stub stub(&rpc_channel);
    janus::proto::Status resp;
    brpc::Controller cntl;
    stub.LinkInstance(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(WARNING) << "LinkInstance RPC failed to " << target_name
                   << ": " << cntl.ErrorText();
      return false;
    }
    if (!resp.ok()) {
      LOG(WARNING) << "LinkInstance RPC returned not-ok from " << target_name;
      return false;
    }
    return true;
  };

  InstanceMetaInfo new_meta_info;
  std::unordered_map<std::string, InstanceMetaInfo> peer_meta_infos;
  {
    std::shared_lock<std::shared_mutex> lock(inst_mutex_);
    auto new_it = instances_.find(instance_name);
    if (new_it != instances_.end()) {
      new_meta_info = new_it->second;
    }
    for (const auto& peer_name : peer_names) {
      auto peer_it = instances_.find(peer_name);
      if (peer_it != instances_.end()) {
        peer_meta_infos.emplace(peer_name, peer_it->second);
      }
    }
  }

  if (!new_meta_info.enable_disagg_pd) {
    LOG(INFO) << "Skip LinkInstance for " << instance_name
              << " because enable_disagg_pd is false";
    return;
  }
  if (new_meta_info.rpc_address.empty()) {
    LOG(WARNING) << "Skip LinkInstance for " << instance_name
                 << " because rpc_address is empty";
    return;
  }

  for (const auto& peer_name : peer_names) {
    if (peer_name == instance_name) {
      // LOG(INFO) << "Skip LinkInstance to self: " << instance_name;
      continue;
    }
    auto peer_it = peer_meta_infos.find(peer_name);
    if (peer_it == peer_meta_infos.end()) {
      continue;
    }
    const auto& peer_meta_info = peer_it->second;
    if (!peer_meta_info.enable_disagg_pd || peer_meta_info.rpc_address.empty()) {
      continue;
    }

    janus::proto::InstanceClusterInfo new_to_peer_req;
    if (!build_link_instance_req(instance_name, new_meta_info, &new_to_peer_req)) {
      LOG(WARNING) << "Skip LinkInstance to peer " << peer_name
                   << " due to invalid new instance metadata";
      continue;
    }

    janus::proto::InstanceClusterInfo peer_to_new_req;
    if (!build_link_instance_req(peer_name, peer_meta_info, &peer_to_new_req)) {
      LOG(WARNING) << "Skip LinkInstance from peer " << peer_name
                   << " due to invalid peer metadata";
      continue;
    }

    // new -> peer
    do_link_instance(peer_name, peer_meta_info.rpc_address, new_to_peer_req);
    // peer -> new
    do_link_instance(instance_name, new_meta_info.rpc_address, peer_to_new_req);
  }
}

}  // namespace janus_service
