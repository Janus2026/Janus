#pragma once

#include "common/macros.h"
#include "disagg_pd.pb.h"
#include "disagg_pd_service.h"
#include "pd_ooc_service_impl.h"

namespace janus {

class PDOOCService : public DisaggPDService {
 public:
  explicit PDOOCService(PDOOCScheduler* scheduler, Engine* engine);
  virtual ~PDOOCService() = default;

  // for decode recv multiple tokens from prefill
  void MultiGenerations(::google::protobuf::RpcController* controller,
                        const proto::DisaggGenerationsRequests* request,
                        proto::Status* response,
                        ::google::protobuf::Closure* done) override;

  void SendPullSignal(::google::protobuf::RpcController* controller,
                      const proto::PullSignal* request,
                      proto::Status* response,
                      ::google::protobuf::Closure* done) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(PDOOCService);
  PDOOCServiceImpl* pd_ooc_service_impl_;  // owned by base class
};

}  // namespace janus
