#include "device_monitor.h"

#include <algorithm>

namespace janus {

void DeviceMonitor::initialize(const std::vector<torch::Device>& devices) {
  for (const auto& device : devices) {
    device_stats_map_[device.index()] = DeviceStats();
    temp_buffer_size_map_[device.index()] = 0;
  }
}

const DeviceStats& DeviceMonitor::get_device_stats(int32_t device_id) const {
  auto it = device_stats_map_.find(device_id);
  if (it == device_stats_map_.end()) {
    LOG(FATAL) << " Fail to find device_id in device_info_map_ : " << device_id;
  }
  return it->second;
}

void DeviceMonitor::set_total_memory(int32_t device_id, int64_t total_memory) {
  device_stats_map_[device_id].total_memory = total_memory;
}

void DeviceMonitor::set_weight_memory(int32_t device_id,
                                      int64_t weights_memory) {
  device_stats_map_[device_id].weights_memory = weights_memory;
}

void DeviceMonitor::set_total_kv_cache_memory(int32_t device_id,
                                              int64_t kv_cache_memory) {
  device_stats_map_[device_id].total_kv_cache_memory = kv_cache_memory;
}

void DeviceMonitor::set_total_activation_memory(int32_t device_id) {
  device_stats_map_[device_id].total_activation_memory =
      device_stats_map_[device_id].total_memory -
      device_stats_map_[device_id].weights_memory -
      device_stats_map_[device_id].total_kv_cache_memory;
}

void DeviceMonitor::monitor_buffer(int32_t device_id, uint64_t buffer_size) {
  temp_buffer_size_map_[device_id] = std::max(
      temp_buffer_size_map_[device_id], static_cast<int64_t>(buffer_size));
}

void DeviceMonitor::update_active_activation_memory(int32_t device_id) {
  device_stats_map_[device_id].active_activation_memory =
      temp_buffer_size_map_[device_id];
  // reset the temp buffer size
  temp_buffer_size_map_[device_id] = 0;
}
}  // namespace janus