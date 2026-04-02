#pragma once

#include <atomic>

namespace janus {

template <class T>
class DoubleBuffer {
 public:
  DoubleBuffer() : index_(0) {}
  virtual ~DoubleBuffer() {}

  T* get_front_value() {
    uint32_t index = front_index();
    return buffers_[index];
  }

  const T* get_front_value() const {
    uint32_t index = front_index();
    return buffers_[index];
  }

  T* get_back_value() {
    uint32_t index = back_index();
    return buffers_[index];
  }

  const T* get_back_value() const {
    uint32_t index = back_index();
    return buffers_[index];
  }

  void set_front_value(T* t) {
    uint32_t index = front_index();
    buffers_[index] = t;
  }

  void set_back_value(T* t) {
    uint32_t index = back_index();
    buffers_[index] = t;
  }

  void swap() {
    uint32_t index = back_index();
    index_.store(index, std::memory_order_release);
  }

 private:
  uint32_t front_index() const {
    return index_.load(std::memory_order_acquire);
  }

  uint32_t back_index() const { return 1 - front_index(); }

  T* buffers_[2] = {nullptr, nullptr};
  std::atomic<uint32_t> index_;
};

enum class DoubleBufferIndex {
  kFront = 0,
  kBack,
};

}  // namespace janus