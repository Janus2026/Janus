#pragma once

#include <google/protobuf/service.h>

#include <functional>

#include "butil/macros.h"

namespace janus {

// RAII: Call Run() of the closure on destruction.
class ClosureGuard {
 public:
  ClosureGuard()
      : _done(nullptr), _before_done([](void*) {}), _after_done([](void*) {}) {}

  // Constructed with a closure which will be Run() inside dtor.
  explicit ClosureGuard(google::protobuf::Closure* done,
                        std::function<void(void*)>&& before_done,
                        std::function<void(void*)>&& after_done)
      : _done(done), _before_done(before_done), _after_done(after_done) {
    _before_done(nullptr);
  }

  // Run internal closure if it's not nullptr.
  ~ClosureGuard() {
    if (_done) {
      _after_done(nullptr);
      _done->Run();
    }
  }

  // Run internal closure if it's not nullptr and set it to `done'.
  void reset(google::protobuf::Closure* done) {
    if (_done) {
      _done->Run();
    }
    _done = done;
  }

  // Return and set internal closure to nullptr.
  google::protobuf::Closure* release() {
    _after_done(nullptr);

    google::protobuf::Closure* const prev_done = _done;
    _done = nullptr;
    return prev_done;
  }

  // True if no closure inside.
  bool empty() const { return _done == nullptr; }

 private:
  // Copying this object makes no sense.
  DISALLOW_COPY_AND_ASSIGN(ClosureGuard);

  google::protobuf::Closure* _done;
  std::function<void(void*)> _before_done;
  std::function<void(void*)> _after_done;
};

}  // namespace janus
