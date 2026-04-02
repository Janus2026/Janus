#include "call.h"

#include "core/common/constants.h"

namespace janus {

Call::Call(brpc::Controller* controller) : controller_(controller) { init(); }

void Call::init() {
  if (controller_->http_request().GetHeader("x-request-id")) {
    x_request_id_ = *controller_->http_request().GetHeader("x-request-id");
  } else if (controller_->http_request().GetHeader("x-ms-client-request-id")) {
    x_request_id_ =
        *controller_->http_request().GetHeader("x-ms-client-request-id");
  }

  if (controller_->http_request().GetHeader("x-request-time")) {
    x_request_time_ = *controller_->http_request().GetHeader("x-request-time");
  } else if (controller_->http_request().GetHeader("x-request-timems")) {
    x_request_time_ =
        *controller_->http_request().GetHeader("x-request-timems");
  }

  init_request_payload();
}

void Call::init_request_payload() {
  const auto infer_content_len =
      controller_->http_request().GetHeader(kInferContentLength);
  const auto content_len =
      controller_->http_request().GetHeader(kContentLength);

  if (infer_content_len == nullptr || content_len == nullptr) return;

  auto infer_len = std::stoul(*infer_content_len);
  auto len = std::stoul(*content_len);

  if (infer_len > len) {
    LOG(ERROR) << " content length is invalid:"
               << " infer content len is " << infer_len
               << " , content length is " << len;
    return;
  }

  controller_->request_attachment().copy_to(
      &request_payload_, len - infer_len, infer_len);
}

}  // namespace janus
