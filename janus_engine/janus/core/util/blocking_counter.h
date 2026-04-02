#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace janus {

class BlockingCounter {
 public:
  BlockingCounter(int initial_count)
      : state_(initial_count << 1), notified_(false) {
    // CHECK(initial_count == 0);
  }
  ~BlockingCounter() = default;

  inline void decrement_count() {
    unsigned int v = state_.fetch_sub(2, std::memory_order_acq_rel) - 2;
    if (v != 1) {
      return;  // either count has not dropped to 0, or waiter is not waiting
    }
    std::lock_guard<std::mutex> lock(mu_);
    // CHECK(!notified_);
    notified_ = true;
    cond_var_.notify_all();
  }

  inline void wait() {
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0) return;
    std::unique_lock<std::mutex> lock(mu_);
    while (!notified_) {
      cond_var_.wait(lock);
    }
  }
  // Wait for the specified time, return false iff the count has not dropped to
  // zero before the timeout expired.
  inline bool wait_for(std::chrono::milliseconds ms) {
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0) return true;
    std::unique_lock<std::mutex> lock(mu_);
    while (!notified_) {
      const std::cv_status status = cond_var_.wait_for(lock, ms);
      if (status == std::cv_status::timeout) {
        return false;
      }
    }
    return true;
  }

 private:
  std::mutex mu_;
  std::condition_variable cond_var_;
  std::atomic<int> state_;  // low bit is waiter flag
  bool notified_;
};

}  // namespace janus
