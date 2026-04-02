#include "rate_limiter.h"

#include <gflags/gflags.h>

#include "common/global_flags.h"
#include "common/metrics.h"

namespace janus {

bool RateLimiter::is_limited() {
  int32_t num_requests =
      num_concurrent_requests_.load(std::memory_order_relaxed);

  // Check if sleeping.
  if (num_requests == kSleeping) {
    return true;
  }

  // Check rate limit.
  if (FLAGS_max_concurrent_requests > 0 &&
      num_requests >= FLAGS_max_concurrent_requests) {
    COUNTER_INC(server_request_total_limit);
    return true;
  }

  num_concurrent_requests_.fetch_add(1, std::memory_order_relaxed);
  GAUGE_SET(num_concurrent_requests,
            num_concurrent_requests_.load(std::memory_order_relaxed));

  return false;
}

void RateLimiter::decrease_one_request() {
  num_concurrent_requests_.fetch_sub(1, std::memory_order_relaxed);
  GAUGE_SET(num_concurrent_requests,
            num_concurrent_requests_.load(std::memory_order_relaxed));
}

void RateLimiter::decrease_requests(size_t decrease_requests_num) {
  num_concurrent_requests_.fetch_sub(decrease_requests_num,
                                     std::memory_order_relaxed);
  GAUGE_SET(num_concurrent_requests,
            num_concurrent_requests_.load(std::memory_order_relaxed));
}

bool RateLimiter::try_set_sleeping() {
  int32_t expected = 0;
  // CAS: only succeed if current value is 0.
  return num_concurrent_requests_.compare_exchange_strong(
      expected, kSleeping, std::memory_order_acq_rel);
}

bool RateLimiter::try_wakeup() {
  int32_t expected = kSleeping;
  // CAS: only succeed if current value is kSleeping.
  return num_concurrent_requests_.compare_exchange_strong(
      expected, 0, std::memory_order_acq_rel);
}

}  // namespace janus
