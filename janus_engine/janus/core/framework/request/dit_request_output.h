#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

#include "core/common/types.h"

namespace janus {

struct DiTGenerationOutput {
  // the index of the sequence in the request.
  size_t index;

  // the generated image in torch tensor format.
  std::string image;

  // the height of the generated image.
  int32_t height;

  // the width of the generated image.
  int32_t width;

  // seed used for image generation.
  int64_t seed;
};

struct DiTRequestOutput {
  DiTRequestOutput() = default;

  DiTRequestOutput(Status&& _status, std::string service_request_id = "")
      : status(std::move(_status)), service_request_id(service_request_id) {}

  void log_request_status() const;

  // the id of the request.
  std::string request_id;

  // the id of the request which generated in janus service.
  std::string service_request_id;

  // the status of the request.
  std::optional<Status> status;

  // the output for each sequence in the request.
  std::vector<DiTGenerationOutput> outputs;

  // whether the request is finished.
  bool finished = false;

  // whether the request is cancelled.
  bool cancelled = false;
};

// callback function for image request output, return true to continue, false to
// stop/cancel
using DiTOutputCallback = std::function<bool(DiTRequestOutput output)>;
// callback function for batch image output, return true to continue, false to
// stop/cancel
using BatchDiTOutputCallback =
    std::function<bool(size_t index, DiTRequestOutput output)>;

}  // namespace janus