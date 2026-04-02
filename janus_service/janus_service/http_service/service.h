#pragma once

#include <brpc/channel.h>

#include <iostream>
#include <mutex>

#include "chat.pb.h"
#include "common/call_data.h"
#include "common/options.h"
#include "common/threadpool.h"
#include "common/types.h"
#include "completion.pb.h"
#include "request/request.h"
#include "request_tracer.h"
#include "janus_http_service.pb.h"

namespace janus_service {

class Scheduler;
class InstanceMgr;
class ClosureGuard;

class JanusHttpServiceImpl : public proto::JanusHttpService {
 public:
  JanusHttpServiceImpl(const Options& options, Scheduler* scheduler);
  ~JanusHttpServiceImpl();

  void Hello(::google::protobuf::RpcController* controller,
             const proto::HttpHelloRequest* request,
             proto::HttpHelloResponse* response,
             ::google::protobuf::Closure* done) override;

  void Completions(::google::protobuf::RpcController* controller,
                   const proto::HttpRequest* request,
                   proto::HttpResponse* response,
                   ::google::protobuf::Closure* done) override;

  void ChatCompletions(::google::protobuf::RpcController* controller,
                       const proto::HttpRequest* request,
                       proto::HttpResponse* response,
                       ::google::protobuf::Closure* done) override;

  void Embeddings(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void Models(::google::protobuf::RpcController* controller,
              const proto::HttpRequest* request,
              proto::HttpResponse* response,
              ::google::protobuf::Closure* done) override;

  void Metrics(::google::protobuf::RpcController* controller,
               const proto::HttpRequest* request,
               proto::HttpResponse* response,
               ::google::protobuf::Closure* done) override;

  void ModelTriggers(::google::protobuf::RpcController* controller,
                     const proto::ModelTriggerRequest* request,
                     proto::HttpResponse* response,
                     ::google::protobuf::Closure* done) override;

 private:
  template <typename T>
  std::shared_ptr<Request> generate_request(T* req_pb,
                                            const std::string& method);

  template <typename T>
  void handle(std::shared_ptr<T> call_data,
              const std::string& req_attachment,
              std::shared_ptr<Request> request,
              const std::string& method);

  void get_serving(const std::string& serving_method,
                   ::google::protobuf::RpcController* controller,
                   const proto::HttpRequest* request,
                   proto::HttpResponse* response,
                   ::google::protobuf::Closure* done);

 private:
  Options options_;

  // not own
  Scheduler* scheduler_;

  bool initialized_ = false;

  std::unique_ptr<RequestTracer> request_tracer_;

  std::unique_ptr<ThreadPool> thread_pool_;

  // In disagg pd mode, we support receive generated token from
  // prefill or from decode directly.
  // 1.
  // [service] ---req---> [prefill] ---req---> [decode]
  // [service] <---first resp--- [prefill] ---first resp---> [decode]
  // [service] <---resp--- [prefill] <---resp--- [decode]
  //
  // 2.
  // [service] ---req---> [prefill] ---req---> [decode]
  // [service] <---first resp-- [prefill] --first resp---> [decode]
  // [service] <---resp-- [decode]
  //
  bool enable_decode_response_to_service_ = false;
};

}  // namespace janus_service
