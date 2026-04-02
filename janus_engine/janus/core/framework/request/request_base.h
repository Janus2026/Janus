#pragma once

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "common.pb.h"
#include "request_state.h"
#include "sequences_group.h"
#include "stopping_checker.h"

namespace janus {

class RequestBase {
 public:
  RequestBase(const std::string& request_id,
              const std::string& x_request_id,
              const std::string& x_request_time,
              const std::string& service_request_id = "",
              const std::string& source_xservice_addr = "")
      : request_id_(request_id),
        x_request_id_(x_request_id),
        x_request_time_(x_request_time),
        service_request_id_(service_request_id),
        source_xservice_addr_(source_xservice_addr),
        created_time_(absl::Now()) {}

  absl::Time created_time() const { return created_time_; }

  const std::string& request_id() const { return request_id_; }

  const std::string& service_request_id() const { return service_request_id_; }

  const std::string& source_xservice_addr() const {
    return source_xservice_addr_;
  }

  const std::string& x_request_id() const { return x_request_id_; }

  const std::string& x_request_time() const { return x_request_time_; }

 protected:
  // request create time
  absl::Time created_time_;

  std::string request_id_;

  std::string service_request_id_;

  std::string source_xservice_addr_;

  // x-request-id header value from client
  std::string x_request_id_;

  // x-request-time header value from client
  std::string x_request_time_;
};

}  // namespace janus
