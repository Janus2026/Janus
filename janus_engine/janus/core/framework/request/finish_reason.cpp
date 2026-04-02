#include "finish_reason.h"

#include <glog/logging.h>

namespace janus {

std::optional<std::string> FinishReason::to_string() {
  switch (value) {
    case Value::NONE:
      return std::nullopt;
    case Value::STOP:
      return "stop";
    case Value::LENGTH:
      return "length";
    case Value::FUNCTION_CALL:
      return "function_call";
    default:
      LOG(WARNING) << "Unknown finish reason: " << static_cast<int>(value);
  }
  return std::nullopt;
}

}  // namespace janus
