#pragma once

#include <brpc/controller.h>

#include <string>

namespace janus {

class Call {
 public:
  Call(brpc::Controller* controller);
  virtual ~Call() = default;

  std::string get_x_request_id() { return x_request_id_; }
  std::string get_x_request_time() { return x_request_time_; }

  std::string take_request_payload() { return std::move(request_payload_); }
  void init_request_payload();

  virtual bool is_disconnected() const = 0;

 protected:
  void init();

 protected:
  brpc::Controller* controller_;

  std::string x_request_id_;
  std::string x_request_time_;

  std::string request_payload_;
};

}  // namespace janus
