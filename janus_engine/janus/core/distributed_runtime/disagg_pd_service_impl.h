#pragma once

#include "disagg_pd.pb.h"
#include "runtime/xservice_client.h"

namespace janus {

class Engine;
class Request;
class DisaggPDScheduler;

// a class to handle disagg_pd requests
class DisaggPDServiceImpl {
 public:
  explicit DisaggPDServiceImpl(DisaggPDScheduler* scheduler, Engine* engine);
  ~DisaggPDServiceImpl() = default;

  virtual void decode_recv_new_requests(const proto::DisaggRequests* request,
                                        proto::DisaggResponses* response);

  virtual void decode_recv_first_generation(
      const proto::DisaggGenerationsRequests* request,
      proto::Status* response);

  virtual void link_instance(const proto::InstanceClusterInfo* request,
                             proto::Status* response);

  virtual void unlink_instance(const proto::InstanceClusterInfo* request,
                               proto::Status* response);

 protected:
  std::shared_ptr<Request> generate_request(const proto::DisaggRequest& req);

  DisaggPDScheduler* scheduler_;  // not owned
  Engine* engine_;                // not owned
  XServiceClient* xservice_client_ = nullptr;
};

}  // namespace janus
