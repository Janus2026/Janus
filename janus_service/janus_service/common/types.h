#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <shared_mutex>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/hash_util.h"
#include "nlohmann/json.hpp"

namespace janus_service {

struct CacheLocations;
using Murmur3KeyCacheMap = std::unordered_map<Murmur3Key,
                                              CacheLocations,
                                              FixedStringKeyHash,
                                              FixedStringKeyEqual>;

struct Routing {
  std::string prefill_name;
  std::string decode_name;

  nlohmann::json serialize_to_json() const {
    nlohmann::json json_val;
    json_val["prefill_name"] = prefill_name;
    json_val["decode_name"] = decode_name;
    return json_val;
  }

  std::string debug_string() const { return serialize_to_json().dump(2); }
};

enum class ErrorCode : int32_t {
  OK = 0,
  INTERNAL_ERROR = 1,
  INSTANCE_EXISTED = 2,
  INSTANCE_NOT_EXISTED = 3,
};

class ConvertErrorCode {
 public:
  static int32_t to_int(ErrorCode code) noexcept {
    return static_cast<int32_t>(code);
  }

  static ErrorCode from_int(int32_t code) noexcept {
    return static_cast<ErrorCode>(code);
  }
};

enum class InstanceType : int8_t {
  DEFAULT = 0,
  // prefill instance
  PREFILL = 1,
  // decode instance
  DECODE = 2,
  // mix instance
  MIX = 3,
};

// SLEEP -> ALLOCATED -> WAKEUP -> DRAINING (skippable) -> SLEEP
enum class ModelState : int32_t {
  WAKEUP = 0,
  SLEEP = 1,
  DRAINING = 2,
  ALLOCATED = 3 // allocated, but not WAKEUP yet
};


struct LoadMetrics {
  LoadMetrics() : waiting_requests_num(0), gpu_cache_usage_perc(0) {};
  LoadMetrics(const uint64_t& waiting_reqs_num, const float& usage)
      : waiting_requests_num(waiting_reqs_num), gpu_cache_usage_perc(usage) {};

  uint64_t waiting_requests_num;
  float gpu_cache_usage_perc;

  nlohmann::json serialize_to_json() const {
    nlohmann::json json_val;
    json_val["waiting_requests_num"] = waiting_requests_num;
    json_val["gpu_cache_usage_perc"] = gpu_cache_usage_perc;
    return json_val;
  }

  std::string debug_string() const { return serialize_to_json().dump(2); }

  bool parse_from_json(const std::string& json_str) {
    try {
      nlohmann::json json_value = nlohmann::json::parse(json_str);

      waiting_requests_num =
          json_value.at("waiting_requests_num").get<uint64_t>();
      gpu_cache_usage_perc = json_value.at("gpu_cache_usage_perc").get<float>();

    } catch (const std::exception& e) {
      LOG(ERROR) << "json str:" << json_str
                 << ", parse to loadmetrics error: " << e.what();
      return false;
    }
    return true;
  }

  bool empty() const { return false; }
};


struct ModelLatencyMetrics {
  int64_t recent_max_ttft = 0;
  int64_t recent_max_tbt = 0;
};

// Record the latency monitoring metrics of the instance over the recent period
struct LatencyMetrics {
  LatencyMetrics() = default;

  // model_id -> metrics
  std::unordered_map<std::string, ModelLatencyMetrics> model_metrics;
};

enum class RequestAction : int32_t {
  SCHEDULE = 0,
  FINISH_PREFILL = 1,
  GENERATE = 2,
  FINISH_DECODE = 3,
  CANCEL = 4,
};

struct ModelRequestMetrics {
  std::condition_variable cv_idle;
  int64_t prefill_request_num = 0;
  int64_t prefill_token_num = 0;
  int64_t decode_request_num = 0;
  int64_t decode_token_num = 0;
};

// Record the request metrics of the instance
struct RequestMetrics {
  
  RequestMetrics() = default;

  // model_id -> metrics
  std::unordered_map<std::string, ModelRequestMetrics> model_metrics;

  // Estimated absolute wall-clock time (ms since epoch) when all queued prefills
  // on this instance will complete. Updated on SCHEDULE/FINISH_PREFILL/CANCEL and
  // corrected by actual PREFILL_DONE feedback.
  int64_t estimated_prefill_done_time = 0;
};

struct InstanceMetaInfo {
 public:
  InstanceMetaInfo() { set_init_timestamp(); }
  InstanceMetaInfo(const std::string& inst_name, const std::string rpc_addr)
      : name(inst_name), rpc_address(rpc_addr) {
    set_init_timestamp();
  }
  InstanceMetaInfo(const std::string& inst_name,
                   const std::string rpc_addr,
                   const InstanceType& inst_type)
      : name(inst_name), rpc_address(rpc_addr), type(inst_type) {
    set_init_timestamp();
  }

  std::string name = "";
  std::string rpc_address = "";
  InstanceType type = InstanceType::DEFAULT;
  std::vector<uint64_t> cluster_ids;
  std::vector<std::string> addrs;
  std::vector<uint64_t> k_cache_ids;
  std::vector<uint64_t> v_cache_ids;
  int32_t dp_size;
  bool enable_disagg_pd = false;
  // Per-worker device addresses for D2D transfer (format: "IP:port")
  std::vector<std::string> device_addrs;
  // P2P addresses for mooncake transfer engine (format: "IP:port")
  std::vector<std::string> p2p_addrs;

  // Per-model DisaggPD RPC addresses: model_id -> rpc_address
  std::unordered_map<std::string, std::string> disagg_pd_rpc_addresses;

  // ttft profiling data per model: model_id -> profiling_data
  std::unordered_map<std::string, std::vector<std::pair<int32_t, double>>> ttft_profiling_data;
  // tpot profiling data per model: model_id -> profiling_data
  std::unordered_map<std::string, std::vector<std::tuple<int32_t, int32_t, double>>> tpot_profiling_data;

  // latest heartbeat timestamp
  uint64_t latest_timestamp = 0;

  uint64_t instance_index = -1;

  // Used to indicate the exact instance type of a MIX type instance currently,
  // only used when the SLO Aware scheduling policy is enabled.
  InstanceType current_type = InstanceType::PREFILL;

  nlohmann::json serialize_to_json() const {
    nlohmann::json json_val;
    json_val["name"] = name;
    json_val["rpc_address"] = rpc_address;
    json_val["type"] = int8_t(type);
    json_val["addrs"] = addrs;
    json_val["cluster_ids"] = cluster_ids;
    json_val["k_cache_ids"] = k_cache_ids;
    json_val["v_cache_ids"] = v_cache_ids;
    json_val["dp_size"] = dp_size;
    json_val["enable_disagg_pd"] = enable_disagg_pd;
    json_val["device_addrs"] = device_addrs;
    json_val["p2p_addrs"] = p2p_addrs;
    
    // Serialize ttft_profiling_data as object with model_id keys
    nlohmann::json ttft_json;
    for (const auto& [model_id, data] : ttft_profiling_data) {
      ttft_json[model_id] = data;
    }
    json_val["ttft_profiling_data"] = ttft_json;
    
    // Serialize tpot_profiling_data as object with model_id keys
    nlohmann::json tpot_json;
    for (const auto& [model_id, data] : tpot_profiling_data) {
      tpot_json[model_id] = data;
    }
    json_val["tpot_profiling_data"] = tpot_json;

    return json_val;
  }

  std::string debug_string() const { return serialize_to_json().dump(2); }

  bool parse_from_json(const std::string& json_str) {
    try {
      nlohmann::json json_value = nlohmann::json::parse(json_str);
      name = json_value.at("name").get<std::string>();
      rpc_address = json_value.at("rpc_address").get<std::string>();
      type = static_cast<InstanceType>(json_value.at("type").get<int8_t>());

      for (const auto& item :
           json_value.at("cluster_ids").get<std::vector<uint64_t>>()) {
        cluster_ids.push_back(item);
      }

      for (const auto& item :
           json_value.at("k_cache_ids").get<std::vector<uint64_t>>()) {
        k_cache_ids.push_back(item);
      }

      for (const auto& item :
           json_value.at("addrs").get<std::vector<std::string>>()) {
        addrs.push_back(item);
      }

      for (const auto& item :
           json_value.at("v_cache_ids").get<std::vector<uint64_t>>()) {
        v_cache_ids.push_back(item);
      }

      dp_size = json_value.at("dp_size").get<int32_t>();

      if (json_value.contains("enable_disagg_pd")) {
        enable_disagg_pd = json_value.at("enable_disagg_pd").get<bool>();
      }

      if (json_value.contains("model_id")) {
        std::string mid = json_value["model_id"].get<std::string>();
        if (!mid.empty() && !rpc_address.empty()) {
          disagg_pd_rpc_addresses[mid] = rpc_address;
        }
      }

      // Parse device_ips and ports, combine into device_addrs ("IP:port")
      if (json_value.contains("device_ips") && json_value.contains("ports")) {
        auto ips = json_value["device_ips"].get<std::vector<std::string>>();
        auto ports = json_value["ports"].get<std::vector<uint16_t>>();
        for (size_t i = 0; i < ips.size() && i < ports.size(); ++i) {
          device_addrs.push_back(ips[i] + ":" + std::to_string(ports[i]));
        }
      }

      // Parse P2P addresses for mooncake transfer engine
      if (json_value.contains("p2p_addrs")) {
        p2p_addrs = json_value["p2p_addrs"].get<std::vector<std::string>>();
      }

      // Parse ttft_profiling_data as object with model_id keys
      if (json_value.contains("ttft_profiling_data")) {
        const auto& ttft_json = json_value.at("ttft_profiling_data");
        for (auto it = ttft_json.begin(); it != ttft_json.end(); ++it) {
          const std::string& model_id = it.key();
          const auto& data_array = it.value();
          std::vector<std::pair<int32_t, double>> model_data;
          for (const auto& item : data_array) {
            if (item.is_array() && item.size() == 2) {
              model_data.emplace_back(item[0].get<int32_t>(), item[1].get<double>());
            }
          }
          ttft_profiling_data[model_id] = std::move(model_data);
        }
      }

      // Parse tpot_profiling_data as object with model_id keys
      if (json_value.contains("tpot_profiling_data")) {
        const auto& tpot_json = json_value.at("tpot_profiling_data");
        for (auto it = tpot_json.begin(); it != tpot_json.end(); ++it) {
          const std::string& model_id = it.key();
          const auto& data_array = it.value();
          std::vector<std::tuple<int32_t, int32_t, double>> model_data;
          for (const auto& item : data_array) {
            if (item.is_array() && item.size() == 3) {
              model_data.emplace_back(
                item[0].get<int32_t>(),
                item[1].get<int32_t>(),
                item[2].get<double>()
              );
            }
          }
          tpot_profiling_data[model_id] = std::move(model_data);
        }
      }

      set_init_timestamp();
    } catch (const std::exception& e) {
      LOG(ERROR) << "json str:" << json_str
                 << ", parse to instancemetainfo error: " << e.what();
      return false;
    }
    return true;
  }

  bool empty() const { return rpc_address == ""; }

 private:
  void set_init_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch())
                            .count();
    latest_timestamp = timestamp_ms;
  }
};

struct CacheLocations {
  std::unordered_set<std::string> hbm_instance_set;
  std::unordered_set<std::string> dram_instance_set;
  std::unordered_set<std::string> ssd_instance_set;

  nlohmann::json serialize_to_json() const {
    nlohmann::json json_val;
    json_val["hbm_instance_set"] = hbm_instance_set;
    json_val["dram_instance_set"] = dram_instance_set;
    json_val["ssd_instance_set"] = ssd_instance_set;
    return json_val;
  }

  std::string debug_string() { return serialize_to_json().dump(2); }

  bool parse_from_json(const std::string& json_str) {
    try {
      nlohmann::json json_value = nlohmann::json::parse(json_str);
      for (const auto& item :
           json_value.at("hbm_instance_set").get<std::vector<std::string>>()) {
        hbm_instance_set.insert(item);
      }

      for (const auto& item :
           json_value.at("dram_instance_set").get<std::vector<std::string>>()) {
        dram_instance_set.insert(item);
      }

      for (const auto& item :
           json_value.at("ssd_instance_set").get<std::vector<std::string>>()) {
        ssd_instance_set.insert(item);
      }

    } catch (const std::exception& e) {
      LOG(ERROR) << "json str:" << json_str
                 << ", parse to cachelocation error: " << e.what();
      return false;
    }
    return true;
  }

  bool empty() const {
    return hbm_instance_set.empty() && dram_instance_set.empty() &&
           ssd_instance_set.empty();
  }
};

/**
 * @brief Records the prefix cache match lengths for different instances on
 * current request
 *
 * This struct stores and manages prefix cache matching information across
 * multiple instances, supporting different storage types (HBM, DRAM, SSD) for
 * match length recording, and tracks information about the best matching
 * instance.
 */
struct OverlapScores {
  // Set of matched instance names
  std::unordered_set<std::string> instances;
  // HBM storage type instance match length mapping (instance name -> match
  // length)
  std::unordered_map<std::string, uint32_t> hbm_instance_score;
  // DRAM storage type instance match length mapping (instance name -> match
  // length)
  std::unordered_map<std::string, uint32_t> dram_instance_score;
  // SSD storage type instance match length mapping (instance name -> match
  // length)
  std::unordered_map<std::string, uint32_t> ssd_instance_score;
  uint32_t max_block_num = 0;
  uint32_t max_matched_block_num = 0;
  std::string max_matched_instance_name = "";

  std::string debug_string() {
    nlohmann::json json_val;
    json_val["instances"] = instances;
    json_val["hbm_instance_score"] = hbm_instance_score;
    json_val["dram_instance_score"] = dram_instance_score;
    json_val["ssd_instance_score"] = ssd_instance_score;
    json_val["max_block_num"] = max_block_num;
    json_val["max_matched_block_num"] = max_matched_block_num;
    json_val["max_matched_instance_name"] = max_matched_instance_name;
    return json_val.dump(2);
  }
};

struct LoadBalanceInfos {
  OverlapScores overlap_scores;
  std::unordered_map<std::string, LoadMetrics> prefill_load_metrics;
  std::unordered_map<std::string, LoadMetrics> decode_load_metrics;
  uint64_t prefill_max_waiting_requests_num = 0;
  uint64_t decode_max_waiting_requests_num = 0;

  std::string debug_string() {
    nlohmann::json json_val;

    json_val["overlap_scores"] =
        nlohmann::json::parse(overlap_scores.debug_string());

    nlohmann::json prefill_json;
    for (auto& [key, metrics] : prefill_load_metrics) {
      prefill_json[key] = nlohmann::json::parse(metrics.debug_string());
    }
    json_val["prefill_load_metrics"] = prefill_json;

    nlohmann::json decode_json;
    for (auto& [key, metrics] : decode_load_metrics) {
      decode_json[key] = nlohmann::json::parse(metrics.debug_string());
    }
    json_val["decode_load_metrics"] = decode_json;

    json_val["prefill_max_waiting_requests_num"] =
        prefill_max_waiting_requests_num;
    json_val["decode_max_waiting_requests_num"] =
        decode_max_waiting_requests_num;

    return json_val.dump(2);
  }
};

// XTensor page size constant (2MB)
static constexpr uint64_t kXTensorPageSizeBytes = 2 * 1024 * 1024;

// Single weight segment in GlobalXtensor memory
struct WeightSegment {
  uint64_t offset;  // Byte offset from GlobalXtensor base
  uint64_t size;    // Segment size in bytes

  uint64_t end() const { return offset + size; }
};

// XTensor memory information for an instance
struct InstanceXTensorInfo {
  // Flag indicating whether valid heartbeat data has been received
  bool is_valid = false;

  // Per-worker free physical pages (index = worker rank)
  std::vector<uint64_t> worker_free_phy_pages;
  // model_id -> segments (each model may have multiple non-contiguous segments)
  std::unordered_map<std::string, std::vector<WeightSegment>> model_weight_segments;
  // Per-worker device addresses for D2D transfer (format: "IP:port")
  std::vector<std::string> device_addrs;
  // P2P addresses for mooncake transfer engine (format: "IP:port")
  std::vector<std::string> p2p_addrs;

  // Get minimum free bytes across all workers (supports TP)
  uint64_t get_min_free_bytes() const {
    if (worker_free_phy_pages.empty()) {
      return 0;
    }
    uint64_t min_pages = *std::min_element(
        worker_free_phy_pages.begin(), worker_free_phy_pages.end());
    return min_pages * kXTensorPageSizeBytes;
  }

  // Get model size from weight segments
  uint64_t get_model_size_bytes(const std::string& model_id) const {
    auto it = model_weight_segments.find(model_id);
    if (it == model_weight_segments.end()) {
      return 0;
    }
    uint64_t total_size = 0;
    for (const auto& seg : it->second) {
      total_size += seg.size;
    }
    return total_size;
  }
};

// D2D wakeup information for device-to-device weight transfer
struct D2DWakeupInfo {
  // Source instance name (for logging/tracking)
  std::string source_instance_name;
  // Remote device addresses for D2D transfer (e.g., "192.168.1.10:40000")
  std::vector<std::string> remote_addrs;
  // Weight segments from source instance (per remote addr, index matches remote_addrs)
  std::vector<std::vector<WeightSegment>> src_weight_segments;

  bool is_valid() const {
    return !remote_addrs.empty() && !src_weight_segments.empty();
  }
};

struct GpuHardwareSpec {
  double hbm_per_gpu_gb = 58.0;
  double compute_sm_per_gpu = 1.0;
  double bandwidth_per_gpu = 1.0;  // HBM-to-SRAM bandwidth capacity (normalized)
};

// Dual-pool scheduling types
enum class PoolType : int8_t { NONE = 0, STEADY = 1, ELASTIC = 2 };

// MixPD instance role tag (orthogonal to PoolType)
enum class InstanceTag : int8_t {
  NONE = 0,
  NORMAL = 1,
  PREFILL = 2,
  DECODE = 3,
};

inline const char* instance_tag_name(InstanceTag tag) {
  switch (tag) {
    case InstanceTag::NONE:    return "NONE";
    case InstanceTag::NORMAL:  return "NORMAL";
    case InstanceTag::PREFILL: return "PREFILL";
    case InstanceTag::DECODE:  return "DECODE";
  }
  return "UNKNOWN";
}

struct ResourceNeeds {
  double hbm_gb = 0.0;
  double compute_sm = 0.0;
  double bandwidth = 0.0;  // HBM-to-SRAM bandwidth utilization [0.0, 1.0]
};

struct SteadyBin {
  std::string instance_name;
  double remaining_hbm_gb;
  double remaining_compute_sm;
  double remaining_bandwidth;  // HBM-to-SRAM bandwidth remaining
  std::unordered_set<std::string> models;  // models loaded on this instance
};

// Function call related types
struct JsonFunction {
  std::string name;
  std::string description;
  nlohmann::json parameters;

  JsonFunction() = default;
  JsonFunction(const std::string& func_name,
               const std::string& desc,
               const nlohmann::json& params)
      : name(func_name), description(desc), parameters(params) {}
};

struct JsonTool {
  std::string type;  // "function"
  JsonFunction function;

  JsonTool() : type("function") {}
  JsonTool(const std::string& tool_type, const JsonFunction& func)
      : type(tool_type), function(func) {}
};

}  // namespace janus_service
