#include "dit_request_output.h"

#include "core/common/metrics.h"

namespace janus {

void DiTRequestOutput::log_request_status() const {
  if (!status.has_value()) {
    return;
  }

  auto code = status.value().code();
  switch (code) {
    case StatusCode::OK:
      COUNTER_INC(request_status_total_ok);
      break;
    case StatusCode::CANCELLED:
      COUNTER_INC(request_status_total_cancelled);
      break;
    case StatusCode::UNKNOWN:
      COUNTER_INC(request_status_total_unknown);
      break;
    case StatusCode::INVALID_ARGUMENT:
      COUNTER_INC(request_status_total_invalid_argument);
      break;
    case StatusCode::DEADLINE_EXCEEDED:
      COUNTER_INC(request_status_total_deadline_exceeded);
      break;
    case StatusCode::RESOURCE_EXHAUSTED:
      COUNTER_INC(request_status_total_resource_exhausted);
      break;
    default:
      COUNTER_INC(request_status_total_unknown);
      break;
  }
}
}  // namespace janus