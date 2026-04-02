#pragma once

#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

namespace janus {

class MMType {
 public:
  enum Value : uint32_t {
    NONE = 0,
    IMAGE = 1 << 0,
    VIDEO = 1 << 1,
    AUDIO = 1 << 2,
    EMBEDDING = 1 << 3
  };

  MMType() = default;
  constexpr MMType(Value v) : value_(v) {}
  constexpr operator Value() const { return value_; }
  explicit operator bool() const = delete;

  constexpr bool operator==(MMType rhs) const { return value_ == rhs.value_; }
  constexpr bool operator!=(MMType rhs) const { return value_ != rhs.value_; }

  constexpr bool operator==(Value v) const { return value_ == v; }
  constexpr bool operator!=(Value v) const { return value_ != v; }

  std::optional<std::string> to_string();

 private:
  Value value_ = Value::NONE;
};

struct ImageMetadata {
  int64_t height = 0;
  int64_t width = 0;
};

struct VideoMetadata {
  double fps = 0.0;              // original fps
  int64_t total_num_frames = 0;  // original frames
  double duration = 0.0;
  double sampled_fps = 0.0;
  torch::Tensor frame_indices;
  std::vector<double> timestamps;
};

struct AudioMetadata {
  int64_t sample_rate = 0;
  int64_t num_channels = 0;
  double duration = 0.0;
};

}  // namespace janus
