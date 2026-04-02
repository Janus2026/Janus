#include "dit_request.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "api_service/call.h"
#include "mm_codec.h"

namespace janus {
DiTRequest::DiTRequest(const std::string& request_id,
                       const std::string& x_request_id,
                       const std::string& x_request_time,
                       const DiTRequestState& state,
                       const std::string& service_request_id,
                       const std::string& source_xservice_addr)
    : RequestBase(request_id,
                  x_request_id,
                  x_request_time,
                  service_request_id,
                  source_xservice_addr),
      state_(state) {}

bool DiTRequest::finished() const { return true; }

void DiTRequest::log_statistic(double total_latency) {
  LOG(INFO) << "x-request-id: " << x_request_id_ << ", "
            << "x-request-time: " << x_request_time_ << ", "
            << "request_id: " << request_id_ << ", "
            << "total_latency: " << total_latency * 1000 << "ms";
}

void DiTRequest::handle_forward_output(torch::Tensor output) {
  int count = state_.generation_params().num_images_per_prompt;
  output_.tensors = torch::chunk(output, count);
}

const DiTRequestOutput DiTRequest::generate_output() {
  DiTRequestOutput output;
  output.request_id = request_id_;
  output.service_request_id = service_request_id_;
  output.status = Status(StatusCode::OK);
  output.finished = finished();
  output.cancelled = false;

  DiTGenerationOutput result;
  result.height = state_.generation_params().height;
  result.width = state_.generation_params().width;
  result.seed = state_.generation_params().seed;

  OpenCVImageEncoder encoder;
  int count = state_.generation_params().num_images_per_prompt;
  for (size_t idx = 0; idx < count; ++idx) {
    torch::Tensor image =
        output_.tensors[idx].squeeze(0).cpu().to(torch::kFloat32).contiguous();
    encoder.encode(image, result.image);
    output.outputs.push_back(result);
  }

  return output;
}

}  // namespace janus
