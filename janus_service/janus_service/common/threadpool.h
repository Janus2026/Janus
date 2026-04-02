#pragma once

#include <functional>
#include <thread>

#include "concurrent_queue.h"

namespace janus_service {

class ThreadPool final {
 public:
  using Task = std::function<void()>;

  // constructors
  ThreadPool() : ThreadPool(10) {}

  // disable copy/move constructor and assignment
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  explicit ThreadPool(size_t num_threads);

  // destructor
  ~ThreadPool();

  // schedule a task to be executed
  void schedule(Task task);

 private:
  void internal_loop();

  std::vector<std::thread> threads_;
  ConcurrentQueue<Task> queue_;
};

}  // namespace janus_service
