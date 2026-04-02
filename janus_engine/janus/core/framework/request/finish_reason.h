#pragma once
#include <cstdint>
#include <optional>
#include <string>

namespace janus {
class FinishReason {
 public:
  enum Value : uint8_t { NONE = 0, STOP = 1, LENGTH, FUNCTION_CALL };

  FinishReason() = default;
  FinishReason(Value v) : value(v) {}
  operator Value() const { return value; }
  explicit operator bool() const = delete;

  bool operator==(FinishReason rhs) const { return value == rhs.value; }
  bool operator!=(FinishReason rhs) const { return value != rhs.value; }

  bool operator==(Value v) const { return value == v; }
  bool operator!=(Value v) const { return value != v; }

  std::optional<std::string> to_string();

 private:
  Value value;
};
}  // namespace janus
