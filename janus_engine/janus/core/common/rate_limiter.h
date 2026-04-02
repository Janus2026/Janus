#pragma once

#include <atomic>

namespace janus {

class RateLimiter final {
 public:
  // Special value indicating sleep state.
  static constexpr int32_t kSleeping = INT32_MIN;

  RateLimiter() = default;

  ~RateLimiter() = default;

  // Returns true if request is rate-limited or sleeping.
  // If not limited and not sleeping, increments the counter.
  bool is_limited();

  void decrease_one_request();

  void decrease_requests(size_t decrease_requests_num);

  int32_t get_num_concurrent_requests() const {
    return num_concurrent_requests_.load(std::memory_order_relaxed);
  }

  // CAS: only succeeds if num_concurrent_requests == 0.
  // Sets to kSleeping on success. Returns true on success.
  bool try_set_sleeping();

  // CAS: only succeeds if num_concurrent_requests == kSleeping.
  // Sets to 0 on success. Returns true on success.
  bool try_wakeup();

  bool is_sleeping() const {
    return num_concurrent_requests_.load(std::memory_order_relaxed) ==
           kSleeping;
  }

 private:
  std::atomic<int32_t> num_concurrent_requests_{0};
};

}  // namespace janus
