#include "pd_ooc_service.h"

#include <brpc/closure_guard.h>
#include <glog/logging.h>

#include "scheduler/pd_ooc_scheduler.h"

namespace janus {

PDOOCService::PDOOCService(PDOOCScheduler* scheduler, Engine* engine) {
  auto pd_ooc_impl = std::make_unique<PDOOCServiceImpl>(scheduler, engine);
  pd_ooc_service_impl_ = pd_ooc_impl.get();
  disagg_pd_service_impl_ = std::move(pd_ooc_impl);
  CHECK(pd_ooc_service_impl_ != nullptr);
}

void PDOOCService::MultiGenerations(
    ::google::protobuf::RpcController* controller,
    const proto::DisaggGenerationsRequests* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  // Receive multiple tokens from Prefill, schedule the request to running queue
  brpc::ClosureGuard done_guard(done);
  pd_ooc_service_impl_->decode_recv_multi_generations(request, response);
}

void PDOOCService::SendPullSignal(::google::protobuf::RpcController* controller,
                                  const proto::PullSignal* request,
                                  proto::Status* response,
                                  ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  pd_ooc_service_impl_->prefill_recv_pull_signal(request, response);
}

}  // namespace janus
