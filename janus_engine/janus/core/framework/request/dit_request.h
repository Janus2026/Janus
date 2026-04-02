#pragma once
#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "common.pb.h"
#include "dit_request_output.h"
#include "dit_request_state.h"
#include "request.h"
#include "runtime/dit_forward_params.h"

namespace janus {

class DiTRequest : public RequestBase {
 public:
  DiTRequest(const std::string& request_id,
             const std::string& x_request_id,
             const std::string& x_request_time,
             const DiTRequestState& state,
             const std::string& service_request_id = "",
             const std::string& source_xservice_addr = "");

  bool finished() const;

  void handle_forward_output(torch::Tensor output);

  const DiTRequestOutput generate_output();

  void log_statistic(double total_latency);

  DiTRequestState& state() { return state_; }

 private:
  DiTRequestState state_;
  DiTForwardOutput output_;
};

}  // namespace janus
