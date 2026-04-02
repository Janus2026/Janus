#include "scheduler/managers/model_instance_mgr.h"
#include <glog/logging.h>
#include <algorithm>
#include <chrono>
#include <thread>
#include <brpc/controller.h>
#include <nlohmann/json.hpp>

namespace janus_service {

ModelInstanceMgr::ModelInstanceMgr(const std::string& model_id)
    : model_id_(model_id) {}

ModelInstanceMgr::~ModelInstanceMgr() {}

void ModelInstanceMgr::add_instance(const std::string& instance_name, const InstanceMetaInfo& info) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  instances_[instance_name] = info;
  
  if (info.type == InstanceType::PREFILL) {
    prefill_index_.push_back(instance_name);
  } else if (info.type == InstanceType::DECODE) {
    decode_index_.push_back(instance_name);
  } else {
    // default/mix
    prefill_index_.push_back(instance_name);
    decode_index_.push_back(instance_name);
  }
}

void ModelInstanceMgr::remove_instance(const std::string& instance_name) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  instances_.erase(instance_name);
  
  // Remove from indices
  auto remove_from_vec = [&](std::vector<std::string>& vec) {
    vec.erase(std::remove(vec.begin(), vec.end(), instance_name), vec.end());
  };
  remove_from_vec(prefill_index_);
  remove_from_vec(decode_index_);
}

void ModelInstanceMgr::update_instance_info(const std::string& instance_name, const InstanceMetaInfo& info) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (instances_.find(instance_name) != instances_.end()) {
    instances_[instance_name] = info;
  }
}

bool ModelInstanceMgr::get_next_instance_pair(Routing* routing) {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  if (prefill_index_.empty() || decode_index_.empty()) {
    return false;
  }

  if (next_prefill_index_ >= prefill_index_.size()) next_prefill_index_ = 0;
  if (next_decode_index_ >= decode_index_.size()) next_decode_index_ = 0;

  std::shared_lock<std::shared_mutex> all_lock(instance_state_all_mutex_);

  // Bounded search to prevent infinite loop when no WAKEUP instances exist
  bool found_prefill = false;
  for (size_t i = 0; i < prefill_index_.size(); ++i) {
    if (instance_states_[prefill_index_[next_prefill_index_]] == ModelState::WAKEUP) {
      found_prefill = true;
      break;
    }
    next_prefill_index_ = (next_prefill_index_ + 1) % prefill_index_.size();
  }
  if (!found_prefill) {
    LOG(WARNING) << "No WAKEUP prefill instance found for model " << model_id_;
    return false;
  }

  bool found_decode = false;
  for (size_t i = 0; i < decode_index_.size(); ++i) {
    if (instance_states_[decode_index_[next_decode_index_]] == ModelState::WAKEUP) {
      found_decode = true;
      break;
    }
    next_decode_index_ = (next_decode_index_ + 1) % decode_index_.size();
  }
  if (!found_decode) {
    LOG(WARNING) << "No WAKEUP decode instance found for model " << model_id_;
    return false;
  }

  routing->prefill_name = prefill_index_[next_prefill_index_];
  routing->decode_name = decode_index_[next_decode_index_];

  next_prefill_index_ = (next_prefill_index_ + 1) % prefill_index_.size();
  next_decode_index_ = (next_decode_index_ + 1) % decode_index_.size();

  return true;
}

void ModelInstanceMgr::flip_prefill_to_decode(const std::string& instance_name) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  
  auto it = std::find(prefill_index_.begin(), prefill_index_.end(), instance_name);
  if (it != prefill_index_.end()) {
    prefill_index_.erase(it);
    decode_index_.push_back(instance_name);
  }
}

void ModelInstanceMgr::flip_decode_to_prefill(const std::string& instance_name) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  
  auto it = std::find(decode_index_.begin(), decode_index_.end(), instance_name);
  if (it != decode_index_.end()) {
    decode_index_.erase(it);
    prefill_index_.push_back(instance_name);
  }
}

std::vector<std::string> ModelInstanceMgr::get_prefill_list() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return prefill_index_;
}

std::vector<std::string> ModelInstanceMgr::get_decode_list() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return decode_index_;
}

bool ModelInstanceMgr::send_http_request(std::shared_ptr<brpc::Channel> channel,
                                         const std::string& uri,
                                         const std::string& request_body,
                                         std::string* error_text) {
  brpc::Controller cntl;
  cntl.http_request().uri() = uri;  // brpc channel already has host:port
  cntl.http_request().set_method(brpc::HTTP_METHOD_POST);
  cntl.http_request().set_content_type("application/json");
  cntl.request_attachment().append(request_body);

  channel->CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);

  if (cntl.Failed()) {
    LOG(ERROR) << "HTTP request failed: " << cntl.ErrorText();
    if (error_text) {
      *error_text = cntl.ErrorText();
    }
    return false;
  }
  return true;
}

bool ModelInstanceMgr::send_model_sleep(const std::string& instance_name, std::shared_ptr<brpc::Channel> channel) {
  std::shared_mutex* instance_state_single_mutex = get_instance_state_single_mutex(instance_name);
  std::unique_lock<std::shared_mutex> single_lock(*instance_state_single_mutex);

  if (instance_states_.count(instance_name) &&
      instance_states_[instance_name] != ModelState::WAKEUP &&
      instance_states_[instance_name] != ModelState::DRAINING) {
    LOG(INFO) << "Model " << model_id_
              << " on " << instance_name
              << " is not suitable for sleep. ModelState : " << static_cast<int32_t>(instance_states_[instance_name]);
    return false;
  }

  nlohmann::json sleep_body;
  sleep_body["model_id"] = model_id_;
  sleep_body["master_status"] = 1;

  static constexpr int kMaxSleepRetries = 10;
  static constexpr int kSleepRetryIntervalMs = 100;

  for (int attempt = 0; attempt < kMaxSleepRetries; ++attempt) {
    std::string error_text;
    if (send_http_request(channel, "/sleep", sleep_body.dump(), &error_text)) {
      LOG(INFO) << "Model " << model_id_ << " on " << instance_name
                << " trigger sleep success.";
      set_model_state(instance_name, ModelState::SLEEP);
      return true;
    }

    if (error_text.find("already sleeping") != std::string::npos) {
      LOG(WARNING) << "Model " << model_id_ << " on " << instance_name
                   << " is already sleeping, treating as sleep success.";
      set_model_state(instance_name, ModelState::SLEEP);
      return true;
    }

    if (error_text.find("Internal Server Error") != std::string::npos) {
      LOG(ERROR) << "Failed to sleep model " << model_id_ << " on "
                 << instance_name << ", non-retryable server error: " << error_text;
      return false;
    }

    LOG(WARNING) << "Failed to sleep model " << model_id_ << " on "
                 << instance_name << ", retry " << attempt + 1
                 << "/" << kMaxSleepRetries << ", error: " << error_text;
    std::this_thread::sleep_for(std::chrono::milliseconds(kSleepRetryIntervalMs));
  }

  LOG(ERROR) << "Failed to sleep model " << model_id_ << " on "
             << instance_name << " after " << kMaxSleepRetries << " retries";
  return false;
}

bool ModelInstanceMgr::send_model_wakeup(const std::string& instance_name, std::shared_ptr<brpc::Channel> channel) {
  std::shared_mutex* instance_state_single_mutex = get_instance_state_single_mutex(instance_name);
  std::unique_lock<std::shared_mutex> single_lock(*instance_state_single_mutex);

  if (instance_states_.count(instance_name) &&
      instance_states_[instance_name] != ModelState::ALLOCATED) {
    LOG(INFO) << "Model " << model_id_
              << " on " << instance_name
              << " is not suitable for wakeup. ModelState : " << static_cast<int32_t>(instance_states_[instance_name]);
    return false;
  }

  nlohmann::json wakeup_body;
  wakeup_body["model_id"] = model_id_;
  wakeup_body["master_status"] = 0;

  static constexpr int kMaxWakeupRetries = 10;
  static constexpr int kWakeupRetryIntervalMs = 100;

  for (int attempt = 0; attempt < kMaxWakeupRetries; ++attempt) {
    std::string error_text;
    if (send_http_request(channel, "/wakeup", wakeup_body.dump(), &error_text)) {
      LOG(INFO) << "Model " << model_id_ << " on " << instance_name
                << " trigger wakeup (H2D) success.";
      set_model_state(instance_name, ModelState::WAKEUP);
      return true;
    }

    if (error_text.find("already awake") != std::string::npos) {
      LOG(WARNING) << "Model " << model_id_ << " on " << instance_name
                   << " is already awake, treating as wakeup success.";
      set_model_state(instance_name, ModelState::WAKEUP);
      return true;
    }

    if (error_text.find("Internal Server Error") != std::string::npos) {
      LOG(ERROR) << "Failed to wakeup model " << model_id_
                 << " on " << instance_name
                 << ", non-retryable server error: " << error_text;
      set_model_state(instance_name, ModelState::SLEEP);
      return false;
    }

    LOG(WARNING) << "Failed to wakeup model " << model_id_
                 << " on " << instance_name << ", retry " << attempt + 1
                 << "/" << kMaxWakeupRetries << ", error: " << error_text;
    std::this_thread::sleep_for(std::chrono::milliseconds(kWakeupRetryIntervalMs));
  }

  LOG(ERROR) << "Failed to wakeup model " << model_id_
             << " on " << instance_name << " after " << kMaxWakeupRetries << " retries";
  set_model_state(instance_name, ModelState::SLEEP);
  return false;
}

bool ModelInstanceMgr::send_model_wakeup_d2d(const std::string& instance_name,
                                              std::shared_ptr<brpc::Channel> channel,
                                              const D2DWakeupInfo& d2d_info) {
  std::shared_mutex* instance_state_single_mutex = get_instance_state_single_mutex(instance_name);
  std::unique_lock<std::shared_mutex> single_lock(*instance_state_single_mutex);

  if (instance_states_.count(instance_name) &&
      instance_states_[instance_name] != ModelState::ALLOCATED) {
    LOG(INFO) << "Model " << model_id_
              << " on " << instance_name
              << " is not suitable for wakeup. ModelState : " << static_cast<int32_t>(instance_states_[instance_name]);
    return false;
  }

  if (!d2d_info.is_valid()) {
    LOG(ERROR) << "Invalid D2D wakeup info for model " << model_id_
               << " on " << instance_name;
    return false;
  }

  nlohmann::json wakeup_body;
  wakeup_body["model_id"] = model_id_;
  wakeup_body["master_status"] = 0;
  wakeup_body["remote_addrs"] = d2d_info.remote_addrs;

  // Build src_weight_segments array (per remote addr)
  nlohmann::json segments_array = nlohmann::json::array();
  for (const auto& segments : d2d_info.src_weight_segments) {
    nlohmann::json segment_list;
    nlohmann::json segs = nlohmann::json::array();
    for (const auto& seg : segments) {
      nlohmann::json seg_json;
      seg_json["offset"] = seg.offset;
      seg_json["size"] = seg.size;
      segs.push_back(seg_json);
    }
    segment_list["segments"] = segs;
    segments_array.push_back(segment_list);
  }
  wakeup_body["src_weight_segments"] = segments_array;

  LOG(INFO) << "Sending D2D wakeup for model " << model_id_
            << " on " << instance_name
            << " from source " << d2d_info.source_instance_name
            << " with " << d2d_info.remote_addrs.size() << " remote addrs";

  static constexpr int kMaxD2DWakeupRetries = 10;
  static constexpr int kD2DWakeupRetryIntervalMs = 100;

  for (int attempt = 0; attempt < kMaxD2DWakeupRetries; ++attempt) {
    std::string error_text;
    if (send_http_request(channel, "/wakeup", wakeup_body.dump(), &error_text)) {
      LOG(INFO) << "Model " << model_id_ << " on " << instance_name
                << " trigger wakeup (D2D) success.";
      set_model_state(instance_name, ModelState::WAKEUP);
      return true;
    }

    if (error_text.find("already awake") != std::string::npos) {
      LOG(WARNING) << "Model " << model_id_ << " on " << instance_name
                   << " is already awake, treating as D2D wakeup success.";
      set_model_state(instance_name, ModelState::WAKEUP);
      return true;
    }

    if (error_text.find("Internal Server Error") != std::string::npos) {
      LOG(ERROR) << "Failed to wakeup model " << model_id_
                 << " on " << instance_name
                 << " with D2D transfer, non-retryable server error: " << error_text;
      set_model_state(instance_name, ModelState::SLEEP);
      return false;
    }

    LOG(WARNING) << "Failed to wakeup model " << model_id_
                 << " on " << instance_name << " with D2D transfer, retry "
                 << attempt + 1 << "/" << kMaxD2DWakeupRetries
                 << ", error: " << error_text;
    std::this_thread::sleep_for(std::chrono::milliseconds(kD2DWakeupRetryIntervalMs));
  }

  LOG(ERROR) << "Failed to wakeup model " << model_id_
             << " on " << instance_name << " with D2D transfer after "
             << kMaxD2DWakeupRetries << " retries";
  set_model_state(instance_name, ModelState::SLEEP);
  return false;
}

bool ModelInstanceMgr::send_link_d2d(std::shared_ptr<brpc::Channel> channel,
                                     const std::vector<std::string>& device_addrs) {
  nlohmann::json body;
  body["model_id"] = model_id_;
  body["device_ips"] = device_addrs;

  LOG(INFO) << "Sending link_d2d for model " << model_id_
            << " with " << device_addrs.size() << " device addrs";

  if (!send_http_request(channel, "/link_d2d", body.dump())) {
    LOG(ERROR) << "Failed to send link_d2d for model " << model_id_;
    return false;
  }
  return true;
}

bool ModelInstanceMgr::send_unlink_d2d(std::shared_ptr<brpc::Channel> channel,
                                       const std::vector<std::string>& device_addrs) {
  nlohmann::json body;
  body["model_id"] = model_id_;
  body["device_ips"] = device_addrs;

  LOG(INFO) << "Sending unlink_d2d for model " << model_id_
            << " with " << device_addrs.size() << " device addrs";

  if (!send_http_request(channel, "/unlink_d2d", body.dump())) {
    LOG(ERROR) << "Failed to send unlink_d2d for model " << model_id_;
    return false;
  }
  return true;
}

void ModelInstanceMgr::link_d2d_bidirectional(
    std::shared_ptr<brpc::Channel> new_channel,
    const std::vector<std::string>& new_device_addrs,
    const std::vector<std::pair<std::shared_ptr<brpc::Channel>,
                                 std::vector<std::string>>>& peers) {
  LOG(INFO) << "Bidirectional D2D linking for model " << model_id_
            << " with " << peers.size() << " peers";

  for (const auto& [peer_channel, peer_addrs] : peers) {
    // new → peer
    if (!send_link_d2d(new_channel, peer_addrs)) {
      LOG(WARNING) << "Failed to link new instance -> peer for model "
                   << model_id_;
    }

    // peer → new
    if (!send_link_d2d(peer_channel, new_device_addrs)) {
      LOG(WARNING) << "Failed to link peer -> new instance for model "
                   << model_id_;
    }
  }
}

bool ModelInstanceMgr::set_model_state(const std::string& instance_name, ModelState new_state) {
  std::unique_lock<std::shared_mutex> all_lock(instance_state_all_mutex_);
  
  if (instance_states_.find(instance_name) == instance_states_.end()) {
    instance_states_[instance_name] = ModelState::SLEEP;
  }

  ModelState current_state = instance_states_[instance_name];
  
  if (new_state == ModelState::WAKEUP) {
    if (current_state == ModelState::ALLOCATED) {
      instance_states_[instance_name] = ModelState::WAKEUP;
      wakeup_count_ += 1;
      return true;
    } else {
      LOG(ERROR) << "ModelInstanceMgr::set_model_state: invalid state transition to WAKEUP from " << static_cast<int32_t>(current_state);
      return false;
    }
  } else if (new_state == ModelState::SLEEP) {
    if (current_state == ModelState::SLEEP) {
      // Already SLEEP, idempotent (e.g. fork_master_and_sleep on newly added instance)
      return true;
    }
    if (current_state == ModelState::DRAINING || current_state == ModelState::WAKEUP) {
      instance_states_[instance_name] = ModelState::SLEEP;
      if (current_state == ModelState::WAKEUP) {
        wakeup_count_ -= 1;
        allocation_count_ -= 1;
      }
      if (current_state == ModelState::DRAINING) {
        // DRAINING was already decremented from wakeup_count_ and allocation_count_
        // when transitioning WAKEUP -> DRAINING
      }
      return true;
    } else if (current_state == ModelState::ALLOCATED) {
      // ALLOCATED -> SLEEP: wakeup failed (e.g. D2D fallback failure), revert allocation
      instance_states_[instance_name] = ModelState::SLEEP;
      allocation_count_ -= 1;
      return true;
    } else {
      LOG(ERROR) << "ModelInstanceMgr::set_model_state: invalid state transition to SLEEP from " << static_cast<int32_t>(current_state);
      return false;
    }
  } else if (new_state == ModelState::ALLOCATED) {
    if (current_state == ModelState::SLEEP) {
      instance_states_[instance_name] = ModelState::ALLOCATED;
      allocation_count_ += 1;
      return true;
    } else {
      LOG(ERROR) << "ModelInstanceMgr::set_model_state: invalid state transition to ALLOCATED from " << static_cast<int32_t>(current_state);
      return false;
    }
  } else if (new_state == ModelState::DRAINING) {
    if (current_state == ModelState::WAKEUP) {
      instance_states_[instance_name] = ModelState::DRAINING;
      wakeup_count_ -= 1;
      allocation_count_ -= 1;
      return true;
    } else {
      LOG(ERROR) << "ModelInstanceMgr::set_model_state: invalid state transition to DRAINING from " << static_cast<int32_t>(current_state);
      return false;
    }
  } else {
    LOG(ERROR) << "ModelInstanceMgr::set_model_state: unsupported state transition. new state: " << static_cast<int32_t>(new_state);
    return false;
  }

  return false;
}

ModelState ModelInstanceMgr::get_model_state(const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> all_lock(instance_state_all_mutex_);
  auto it = instance_states_.find(instance_name);
  if (it == instance_states_.end()) {
    return ModelState::SLEEP;
  }
  return it->second;
}

bool ModelInstanceMgr::is_model_waking_up() {
  std::shared_lock<std::shared_mutex> all_lock(instance_state_all_mutex_);
  return allocation_count_ > 0;
}

int32_t ModelInstanceMgr::get_wakeup_count() {
  std::shared_lock<std::shared_mutex> all_lock(instance_state_all_mutex_);
  return wakeup_count_;
}

int32_t ModelInstanceMgr::get_allocation_count() {
  std::shared_lock<std::shared_mutex> all_lock(instance_state_all_mutex_);
  return allocation_count_;
}

std::vector<std::string> ModelInstanceMgr::get_awake_instances() {
  std::shared_lock<std::shared_mutex> all_lock(instance_state_all_mutex_);
  std::vector<std::string> awake_instances;
  for (const auto& pair : instance_states_) {
    if (pair.second == ModelState::WAKEUP) {
      awake_instances.push_back(pair.first);
    }
  }
  return awake_instances;
}

std::vector<std::string> ModelInstanceMgr::get_active_instances() {
  std::shared_lock<std::shared_mutex> all_lock(instance_state_all_mutex_);
  std::vector<std::string> active;
  for (const auto& pair : instance_states_) {
    if (pair.second == ModelState::WAKEUP || pair.second == ModelState::ALLOCATED) {
      active.push_back(pair.first);
    }
  }
  return active;
}

std::vector<std::string> ModelInstanceMgr::get_all_instance_names() {
  std::shared_lock<std::shared_mutex> all_lock(instance_state_all_mutex_);
  std::vector<std::string> names;
  names.reserve(instance_states_.size());
  for (const auto& pair : instance_states_) {
    names.push_back(pair.first);
  }
  return names;
}

std::vector<std::string> ModelInstanceMgr::get_unlocked_instances() {
  std::shared_lock<std::shared_mutex> all_lock(instance_state_all_mutex_);
  std::lock_guard<std::mutex> d2d_lock(d2d_ref_mutex_);
  std::vector<std::string> unlocked_instances;
  for (const auto& pair : instance_states_) {
    if (pair.second == ModelState::WAKEUP) {
      // Only include if not locked (ref_count == 0)
      if (d2d_ref_counts_.count(pair.first) == 0 ||
          d2d_ref_counts_[pair.first] == 0) {
        unlocked_instances.push_back(pair.first);
      }
    }
  }
  return unlocked_instances;
}

std::vector<std::string> ModelInstanceMgr::get_awake_instances_and_lock() {
  std::shared_lock<std::shared_mutex> all_lock(instance_state_all_mutex_);
  std::lock_guard<std::mutex> d2d_lock(d2d_ref_mutex_);
  std::vector<std::string> awake_instances;
  for (const auto& pair : instance_states_) {
    if (pair.second == ModelState::WAKEUP) {
      awake_instances.push_back(pair.first);
      d2d_ref_counts_[pair.first]++;
    }
  }
  LOG(INFO) << "Model " << model_id_ << " get_awake_instances_and_lock: locked "
            << awake_instances.size() << " instances";
  return awake_instances;
}

void ModelInstanceMgr::update_model_heat(int64_t token_count, int64_t input_len) {
  std::lock_guard<std::mutex> heat_lock(model_heat_mutex_);
  prune_model_heat_locked();
  model_heat_records_.push_back(
      {std::chrono::steady_clock::now(), token_count, input_len});
  model_heat_ += token_count;
  total_input_len_ += input_len;
  total_input_len2_ += input_len * input_len;
  request_count_ += 1;
}

int64_t ModelInstanceMgr::get_model_heat() {
  std::lock_guard<std::mutex> heat_lock(model_heat_mutex_);
  prune_model_heat_locked();
  return model_heat_;
}

double ModelInstanceMgr::get_avg_token_rate(int window_seconds) {
  std::lock_guard<std::mutex> heat_lock(model_heat_mutex_);
  if (window_seconds <= 0) return 0.0;
  auto now = std::chrono::steady_clock::now();
  int64_t total_tokens = 0;
  for (const auto& record : model_heat_records_) {
    auto age = std::chrono::duration_cast<std::chrono::seconds>(
        now - record.timestamp).count();
    if (age <= window_seconds) {
      total_tokens += record.token_count;
    }
  }
  return static_cast<double>(total_tokens) / window_seconds;
}

ModelInstanceMgr::TrafficStats ModelInstanceMgr::get_traffic_stats() {
  std::lock_guard<std::mutex> heat_lock(model_heat_mutex_);
  prune_model_heat_locked();

  TrafficStats stats;
  if (kModelHeatRetentionSeconds > 0) {
    stats.token_rate =
        static_cast<double>(model_heat_) / kModelHeatRetentionSeconds;
  }
  if (request_count_ > 0) {
    stats.avg_input_len =
        static_cast<double>(total_input_len_) / request_count_;
    stats.avg_input_len2 =
        static_cast<double>(total_input_len2_) / request_count_;
  }
  // avg_output_len uses default (20.0) — actual tracking TBD
  return stats;
}

void ModelInstanceMgr::prune_model_heat_locked() {
  auto& records = model_heat_records_;
  auto now = std::chrono::steady_clock::now();
  while (!records.empty()) {
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - records.front().timestamp).count();
    if (duration > kModelHeatRetentionSeconds) {
      model_heat_ -= records.front().token_count;
      int64_t il = records.front().input_len;
      total_input_len_ -= il;
      total_input_len2_ -= il * il;
      request_count_ -= 1;
      records.pop_front();
    } else {
      break;
    }
  }
  if (model_heat_ < 0) {
    LOG(WARNING) << "ModelInstanceMgr::prune_model_heat_locked: model_heat_ < 0, reset to 0.";
    model_heat_ = 0;
  }
  if (request_count_ < 0) {
    request_count_ = 0;
    total_input_len_ = 0;
    total_input_len2_ = 0;
  }
}

void ModelInstanceMgr::auto_flipping(const std::unordered_map<std::string, LatencyMetrics>& latency_metrics) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  std::shared_lock<std::shared_mutex> state_lock(instance_state_all_mutex_);

  int prefill_count = prefill_index_.size();
  int decode_count = decode_index_.size();

  for (const auto& pair : instance_states_) {
    if (pair.second != ModelState::WAKEUP) continue;
    
    std::string instance_name = pair.first;
    if (instances_.find(instance_name) == instances_.end()) continue;
    
    const auto& info = instances_[instance_name];
    if (info.type != InstanceType::MIX) continue;

    auto it = latency_metrics.find(instance_name);
    if (it == latency_metrics.end()) continue;
    
    auto model_it = it->second.model_metrics.find(model_id_);
    if (model_it == it->second.model_metrics.end()) continue;
    
    const auto& metrics = model_it->second;
    
    bool is_prefill = std::find(prefill_index_.begin(), prefill_index_.end(), instance_name) != prefill_index_.end();
    bool is_decode = std::find(decode_index_.begin(), decode_index_.end(), instance_name) != decode_index_.end();
    
    const double HIGH_TTFT_MS = 2000.0;
    const double HIGH_TBT_MS = 100.0;
    
    if (metrics.recent_max_ttft > HIGH_TTFT_MS) {
      if (is_decode && !is_prefill) {
        auto d_it = std::find(decode_index_.begin(), decode_index_.end(), instance_name);
        if (d_it != decode_index_.end()) {
          decode_index_.erase(d_it);
          prefill_index_.push_back(instance_name);
          
          LOG(INFO) << "Model " << model_id_ << " on " << instance_name 
                    << ": High TTFT (" << metrics.recent_max_ttft 
                    << "), flipping DECODE -> PREFILL. New counts: P=" 
                    << prefill_index_.size() << ", D=" << decode_index_.size();
                    
          prefill_count++;
          decode_count--;
        }
      }
    } else if (metrics.recent_max_tbt > HIGH_TBT_MS) {
      if (is_prefill && !is_decode) {
        if (prefill_count > 1) {
          auto p_it = std::find(prefill_index_.begin(), prefill_index_.end(), instance_name);
          if (p_it != prefill_index_.end()) {
            prefill_index_.erase(p_it);
            decode_index_.push_back(instance_name);
            
            LOG(INFO) << "Model " << model_id_ << " on " << instance_name 
                      << ": High TBT (" << metrics.recent_max_tbt 
                      << "), flipping PREFILL -> DECODE. New counts: P=" 
                      << prefill_index_.size() << ", D=" << decode_index_.size();

            prefill_count--;
            decode_count++;
          }
        }
      }
    }
  }

  // Safety check for indices
  if (next_prefill_index_ >= prefill_index_.size()) next_prefill_index_ = 0;
  if (next_decode_index_ >= decode_index_.size()) next_decode_index_ = 0;
}

std::shared_mutex* ModelInstanceMgr::get_instance_state_single_mutex(const std::string& instance_name) {
  std::unique_lock<std::shared_mutex> all_lock(instance_state_all_mutex_);
  if (instance_state_single_mutexes_.find(instance_name) == instance_state_single_mutexes_.end()) {
    instance_state_single_mutexes_[instance_name] = std::make_unique<std::shared_mutex>();
  }
  return instance_state_single_mutexes_[instance_name].get();
}

void ModelInstanceMgr::acquire_d2d_lock(const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(d2d_ref_mutex_);
  d2d_ref_counts_[instance_name]++;
  LOG(INFO) << "Model " << model_id_ << " on " << instance_name
            << " D2D lock acquired, ref_count=" << d2d_ref_counts_[instance_name];
}

void ModelInstanceMgr::release_d2d_lock(const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(d2d_ref_mutex_);
  if (d2d_ref_counts_.count(instance_name) && d2d_ref_counts_[instance_name] > 0) {
    d2d_ref_counts_[instance_name]--;
    LOG(INFO) << "Model " << model_id_ << " on " << instance_name
              << " D2D lock released, ref_count=" << d2d_ref_counts_[instance_name];
  } else {
    LOG(WARNING) << "Model " << model_id_ << " on " << instance_name
                 << " D2D lock release called but ref_count is already 0";
  }
}

void ModelInstanceMgr::release_d2d_locks(const std::vector<std::string>& instance_names) {
  std::lock_guard<std::mutex> lock(d2d_ref_mutex_);
  for (const auto& instance_name : instance_names) {
    if (d2d_ref_counts_.count(instance_name) && d2d_ref_counts_[instance_name] > 0) {
      d2d_ref_counts_[instance_name]--;
    }
  }
  LOG(INFO) << "Model " << model_id_ << " batch released D2D locks for "
            << instance_names.size() << " instances";
}

bool ModelInstanceMgr::can_sleep(const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(d2d_ref_mutex_);
  if (d2d_ref_counts_.count(instance_name) == 0) {
    return true;
  }
  return d2d_ref_counts_[instance_name] == 0;
}

int32_t ModelInstanceMgr::get_d2d_ref_count(const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(d2d_ref_mutex_);
  if (d2d_ref_counts_.count(instance_name) == 0) {
    return 0;
  }
  return d2d_ref_counts_[instance_name];
}

}  // namespace janus_service