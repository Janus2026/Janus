#pragma once

#include "disagg_pd.pb.h"
#include "disagg_pd_service_impl.h"

namespace janus {

class Engine;
class Request;
class PDOOCScheduler;

// a class to handle disagg_pd OOC requests
class PDOOCServiceImpl final : public DisaggPDServiceImpl {
 public:
  explicit PDOOCServiceImpl(PDOOCScheduler* scheduler, Engine* engine);
  ~PDOOCServiceImpl() = default;

  virtual void decode_recv_multi_generations(
      const proto::DisaggGenerationsRequests* request,
      proto::Status* response);

  virtual void prefill_recv_pull_signal(const proto::PullSignal* request,
                                        proto::Status* response);

 private:
  PDOOCScheduler* pd_ooc_scheduler_;  // not owned
};

}  // namespace janus
