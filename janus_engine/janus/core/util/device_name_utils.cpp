#include "device_name_utils.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <glog/logging.h>

#include "platform/device.h"

namespace janus {

std::vector<torch::Device> DeviceNameUtils::parse_devices(
    const std::string& device_str) {
  std::vector<torch::Device> devices;
  if (device_str == "auto" || device_str.empty()) {
    // use all available devices if any
    const auto num_devices = Device::device_count();
    if (num_devices == 0) {
      LOG(INFO) << "no devices found, using cpu.";
      return {torch::kCPU};
    }
    devices.reserve(num_devices);
    for (int i = 0; i < num_devices; ++i) {
      std::string device_name = Device::type_str() + ":" + std::to_string(i);
      devices.emplace_back(torch::Device(device_name));
    }
    return devices;
  }

  // parse device string
  const std::vector<std::string> device_strs = absl::StrSplit(device_str, ',');
  std::unordered_set<torch::DeviceType> device_types;
  devices.reserve(device_strs.size());
  for (const auto& device_str : device_strs) {
    std::vector<std::string> parts = absl::StrSplit(device_str, ':');
    CHECK(parts.size() == 2) << "Invalid device string format: " << device_str;
    CHECK(parts[0] == Device::type_str())
        << "Unsupported device type: " << parts[0];

    int device_index;
    CHECK(absl::SimpleAtoi(parts[1], &device_index))
        << "Invalid device index: " << parts[1];

    devices.emplace_back(Device::type_torch(), device_index);
    device_types.insert(devices.back().type());
  }
  CHECK(!devices.empty()) << "No devices specified.";
  CHECK(device_types.size() == 1)
      << "All devices must be of the same type. Got: " << device_str;
  return devices;
}

}  // namespace janus
