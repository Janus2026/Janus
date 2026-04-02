#pragma once

#include <cstdint>

namespace janus {

// REC model execution step constants
constexpr size_t kRecDecodeSteps = 2;  // Number of decode iterations
constexpr size_t kRecTotalSteps = kRecDecodeSteps + 1;  // 1 prefill + N decode

enum class RecType : uint8_t {
  kNone = 0,
  kOneRec = 1,
  kLlmRec = 2,
};

}  // namespace janus
