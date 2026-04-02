#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "macros.h"
#include "types.h"

namespace janus {

class DeviceMonitor {
 public:
  static DeviceMonitor& get_instance() {
    static DeviceMonitor instance;
    return instance;
  }

  void initialize(const std::vector<torch::Device>& devices);
  const DeviceStats& get_device_stats(int32_t device_id) const;
  void set_total_memory(int32_t device_id, int64_t total_memory);
  void set_weight_memory(int32_t device_id, int64_t weights_memory);
  void set_total_kv_cache_memory(int32_t device_id, int64_t kv_cache_memory);
  void set_total_activation_memory(int32_t device_id);
  void monitor_buffer(int32_t device_id, uint64_t buffer_size);
  void update_active_activation_memory(int32_t device_id);

 private:
  DeviceMonitor() = default;
  ~DeviceMonitor() = default;
  DISALLOW_COPY_AND_ASSIGN(DeviceMonitor);

  std::map<int32_t, DeviceStats> device_stats_map_;
  std::map<int32_t, int64_t> temp_buffer_size_map_;
};
}  // namespace janus