#pragma once

#include <c10/core/Device.h>

#include <memory>
#include <mutex>
#include <unordered_map>

#include "common/macros.h"

namespace janus {
namespace npu {

// Device-level mutex manager for protecting ACL Graph capture operations
// Prevents prepare_work_before_execute and capture from executing
// simultaneously to avoid synchronization conflicts during capture
class DeviceCaptureLock {
 public:
  // Get singleton instance
  static DeviceCaptureLock& get_instance() {
    static DeviceCaptureLock instance;
    return instance;
  }

  // Get mutex for a specific device
  // Creates a new mutex if one doesn't exist for the device
  std::mutex& get_lock(c10::DeviceIndex device_index) {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    auto it = locks_.find(device_index);
    if (it == locks_.end()) {
      locks_[device_index] = std::make_unique<std::mutex>();
      return *locks_[device_index];
    }
    return *it->second;
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(DeviceCaptureLock);
  DeviceCaptureLock() = default;
  ~DeviceCaptureLock() = default;

  // Map from device index to mutex
  std::unordered_map<c10::DeviceIndex, std::unique_ptr<std::mutex>> locks_;
  // Mutex to protect the map itself
  std::mutex map_mutex_;
};

}  // namespace npu
}  // namespace janus
