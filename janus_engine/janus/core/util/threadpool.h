#pragma once
#include <folly/Function.h>

#include <thread>

#include "concurrent_queue.h"
#include "util/blocking_counter.h"
namespace janus {

class ThreadPool final {
 public:
  // a runnable is an object intended to be executed by the threadpool
  // it must be invokable with no arguments and return void.
  using Runnable = folly::Function<void()>;

  // constructors
  ThreadPool() : ThreadPool(1) {}
  // destructor
  ~ThreadPool();

  // disable copy/move constructor and assignment
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  explicit ThreadPool(size_t num_threads);
  explicit ThreadPool(size_t num_threads, Runnable init_func);

  // schedule a runnable to be executed
  int32_t schedule(Runnable runnable);

  void schedule_with_tid(Runnable runnable, size_t tid);

  bool empty() {
    return std::all_of(queues_.begin(), queues_.end(), [](auto& queue) {
      return queue.empty();
    });
  }

  size_t size() { return threads_.size(); }

 private:
  void internal_loop(size_t tid,
                     Runnable* init_func,
                     BlockingCounter* block_counter);

  std::vector<std::thread> threads_;
  std::vector<ConcurrentQueue<Runnable>> queues_;

  std::atomic<size_t> index_{0};
};

}  // namespace janus
