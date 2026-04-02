#pragma once

#include <brpc/controller.h>
#include <butil/iobuf.h>
#include <glog/logging.h>
#include <json2pb/pb_to_json.h>

#include <functional>
#include <string>

#include "chat.pb.h"
#include "completion.pb.h"

namespace janus_service {

// Interface for the classes that are used to handle grpc requests.
class CallData {
 public:
  virtual ~CallData() = default;

  // returns true if the rpc is ok and the call data is not finished
  // returns false if the call data is finished and can be deleted
  virtual bool proceed(bool rpc_ok) = 0;

  void get_x_request_id(std::string& x_request_id, brpc::Controller* ctrl) {
    x_request_id = "";
    if (ctrl->http_request().GetHeader("x-request-id")) {
      x_request_id = *ctrl->http_request().GetHeader("x-request-id");
    } else if (ctrl->http_request().GetHeader("x-ms-client-request-id")) {
      x_request_id = *ctrl->http_request().GetHeader("x-ms-client-request-id");
    }
    return;
  }

  void get_x_request_time(std::string& x_request_time, brpc::Controller* ctrl) {
    x_request_time = "";
    if (ctrl->http_request().GetHeader("x-request-time")) {
      x_request_time = *ctrl->http_request().GetHeader("x-request-time");
    } else if (ctrl->http_request().GetHeader("x-request-timems")) {
      x_request_time = *ctrl->http_request().GetHeader("x-request-timems");
    }
    return;
  }

 public:
  std::string x_request_id;
  std::string x_request_time;
};

template <typename Request, typename Response>
class StreamCallData : public CallData {
 public:
  StreamCallData(
      brpc::Controller* controller,
      bool stream,
      ::google::protobuf::Closure* done,
      Response* response,
      std::function<void(const std::string&)> trace_callback = nullptr)
      : controller_(controller),
        done_(done),
        response_(response),
        trace_callback_(std::move(trace_callback)) {
    stream_ = stream;
    get_x_request_id(x_request_id, controller_);
    get_x_request_time(x_request_time, controller_);

    if (stream_) {
      pa_ = controller_->CreateProgressiveAttachment();

      controller_->http_response().set_content_type("text/event-stream");
      controller_->http_response().set_status_code(200);
      controller_->http_response().SetHeader("Connection", "keep-alive");
      controller_->http_response().SetHeader("Cache-Control", "no-cache");
      // Done Run first for steam response
      done_->Run();

    } else {
      controller_->http_response().SetHeader("Content-Type",
                                             "text/javascript; charset=utf-8");
    }

    json_options_.bytes_to_base64 = false;
    json_options_.jsonify_empty_array = true;
  }

  ~StreamCallData() {
    // For non stream response, call brpc done Run
    if (!stream_) {
      done_->Run();
    }
  }

  bool proceed(bool rpc_ok) override { return true; }

  // For non stream response
  bool write_and_finish(const std::string& attachment /*json string*/) {
    if (trace_callback_) trace_callback_(attachment);
    controller_->response_attachment() = attachment;
    return true;
  }

  bool write_and_finish(Response& response) {
    butil::IOBufAsZeroCopyOutputStream json_output(
        &controller_->response_attachment());
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      return finish_with_error(err_msg);
    }

    if (trace_callback_) {
      std::string str;
      controller_->response_attachment().copy_to(&str);
      trace_callback_(str);
    }

    return true;
  }

  // For non stream response
  bool finish_with_error(const std::string& error_message,
                         int http_status_code = 500) {
    if (!stream_) {
      controller_->http_response().set_status_code(http_status_code);
      controller_->SetFailed(error_message);
    } else {
      butil::IOBuf buf;
      buf.append(error_message);
      pa_->Write(buf);
    }

    return true;
  }

  // For stream response
  bool write(const butil::IOBuf& attachment_iobuf) {
    if (trace_callback_) {
      std::string str;
      attachment_iobuf.copy_to(&str);
      trace_callback_(str);
    }
    pa_->Write(attachment_iobuf);
    return true;
  }

  // For stream response
  bool write(const std::string& attachment) {
    if (trace_callback_) trace_callback_(attachment);
    butil::IOBuf buf;
    buf.append(attachment);
    pa_->Write(buf);
    if (attachment.find("data: [DONE]") != std::string::npos) {
      finished_ = true;
    }

    return true;
  }

  bool write(Response& response) {
    butil::IOBuf buf;
    buf.append("data: ");
    butil::IOBufAsZeroCopyOutputStream json_output(&buf);
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      LOG(ERROR) << "Failed to convert proto to json: " << err_msg;
      return false;
    }
    buf.append("\n\n");

    if (trace_callback_) {
      std::string str;
      buf.copy_to(&str);
      trace_callback_(str);
    }

    pa_->Write(buf);
    return true;
  }

  bool finish() {
    butil::IOBuf buf;
    buf.append("data: [DONE]\n\n");

    pa_->Write(buf);
    return true;
  }

  Response& response() { return *response_; }
  ::google::protobuf::Closure* done() { return done_; }
  bool finished() { return finished_; }

 private:
  brpc::Controller* controller_;
  ::google::protobuf::Closure* done_;

  Response* response_;

  bool stream_ = false;
  butil::intrusive_ptr<brpc::ProgressiveAttachment> pa_;

  bool finished_ = false;
  json2pb::Pb2JsonOptions json_options_;
  std::function<void(const std::string&)> trace_callback_;
};

using CompletionCallData = StreamCallData<llm::proto::CompletionRequest,
                                          llm::proto::CompletionResponse>;

using ChatCallData =
    StreamCallData<llm::proto::ChatRequest, llm::proto::ChatResponse>;

}  // namespace janus_service