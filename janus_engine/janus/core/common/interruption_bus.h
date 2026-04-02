#pragma once

#include <functional>
#include <vector>

namespace janus {
class ForwardInterruptedException : public std::exception {};

class InterruptionBus {
 public:
  void subscribe(std::function<void(bool)> func) { observers_.push_back(func); }

  void publish(bool interruption) {
    for (auto it = observers_.cbegin(); it != observers_.cend(); ++it) {
      const auto& observer = *it;
      observer(interruption);
    }
  }

  static InterruptionBus& get_instance() {
    static InterruptionBus instance;
    return instance;
  }

 private:
  std::vector<std::function<void(bool)>> observers_;
};
}  // namespace janus