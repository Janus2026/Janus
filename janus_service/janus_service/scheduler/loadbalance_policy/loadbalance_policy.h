#pragma once

#include "common/types.h"
#include "request/request.h"
#include "scheduler/managers/instance_mgr.h"

namespace janus_service {

class LoadBalancePolicy {
 public:
  LoadBalancePolicy(std::shared_ptr<InstanceMgr> instance_mgr)
      : instance_mgr_(instance_mgr) {}

  virtual ~LoadBalancePolicy() = default;

  virtual bool select_instances_pair(std::shared_ptr<Request> request) = 0;

  // Called when prefill completes on an instance; LstImhPolicy overrides to
  // wake the coordinator so it can re-evaluate availability.
  virtual void on_prefill_done(const std::string& instance_name) {}

  // Called at shutdown; LstImhPolicy overrides to unblock threads waiting
  // inside select_instances_pair().
  virtual void shutdown() {}

 protected:
  std::shared_ptr<InstanceMgr> instance_mgr_;
};

}  // namespace janus_service
