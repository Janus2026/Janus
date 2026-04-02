#pragma once

#include "common/macros.h"
#include "loadbalance_policy.h"

namespace janus_service {

class RoundRobin final : public LoadBalancePolicy {
 public:
  RoundRobin(std::shared_ptr<InstanceMgr> instance_mgr)
      : LoadBalancePolicy(instance_mgr) {};

  virtual ~RoundRobin() = default;

  bool select_instances_pair(std::shared_ptr<Request> request) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(RoundRobin);
};

}  // namespace janus_service
