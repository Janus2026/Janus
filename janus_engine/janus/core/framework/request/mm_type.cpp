#include "mm_type.h"

namespace janus {

std::optional<std::string> MMType::to_string() {
  switch (value_) {
    case Value::NONE:
      return std::nullopt;
    case Value::IMAGE:
      return "image";
    case Value::VIDEO:
      return "video";
    case Value::AUDIO:
      return "audio";
    case Value::EMBEDDING:
      return "embedding";
    default:
      LOG(WARNING) << "Unknown mm type: " << static_cast<int>(value_);
  }
  return std::nullopt;
}

}  // namespace janus
