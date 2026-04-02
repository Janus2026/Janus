#pragma once

#include <torch/torch.h>

#include <vector>

namespace janus {

class DeviceNameUtils {
 public:
  static std::vector<torch::Device> parse_devices(
      const std::string& device_str);

  template <typename T>
  static std::string to_string(const std::vector<T>& items) {
    std::stringstream ss;
    for (size_t i = 0; i < items.size(); ++i) {
      const auto& item = items[i];
      if (i == 0) {
        ss << item;
      } else {
        ss << "," << item;
      }
    }
    return ss.str();
  }
};

}  // namespace janus
