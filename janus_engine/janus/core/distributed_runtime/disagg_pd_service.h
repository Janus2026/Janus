#pragma once

#include "common/macros.h"
#include "disagg_pd.pb.h"
#include "disagg_pd_service_impl.h"

namespace janus {

class DisaggPDService : public proto::DisaggPDService {
 public:
  explicit DisaggPDService(DisaggPDScheduler* scheduler, Engine* engine);
  explicit DisaggPDService() {}
  virtual ~DisaggPDService() = default;

  // for decode recv prefill request
  void AddNewRequests(::google::protobuf::RpcController* controller,
                      const proto::DisaggRequests* request,
                      proto::DisaggResponses* response,
                      ::google::protobuf::Closure* done) override;

  // for decode recv first token from prefill
  void FirstGeneration(::google::protobuf::RpcController* controller,
                       const proto::DisaggGenerationsRequests* request,
                       proto::Status* response,
                       ::google::protobuf::Closure* done) override;

  // for decode recv multiple tokens from prefill
  virtual void MultiGenerations(::google::protobuf::RpcController* controller,
                                const proto::DisaggGenerationsRequests* request,
                                proto::Status* response,
                                ::google::protobuf::Closure* done) override;

  virtual void SendPullSignal(::google::protobuf::RpcController* controller,
                              const proto::PullSignal* request,
                              proto::Status* response,
                              ::google::protobuf::Closure* done) override;

  virtual void LinkInstance(::google::protobuf::RpcController* controller,
                            const proto::InstanceClusterInfo* request,
                            proto::Status* response,
                            ::google::protobuf::Closure* done) override;

  virtual void UnlinkInstance(::google::protobuf::RpcController* controller,
                              const proto::InstanceClusterInfo* request,
                              proto::Status* response,
                              ::google::protobuf::Closure* done) override;

 protected:
  std::unique_ptr<DisaggPDServiceImpl> disagg_pd_service_impl_;

 private:
  DISALLOW_COPY_AND_ASSIGN(DisaggPDService);
};

}  // namespace janus
