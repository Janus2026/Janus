#pragma once

#include "common/options.h"
#include "common/types.h"
#include "loadbalance_policy.h"

namespace janus_service {

class SloAwarePolicy final : public LoadBalancePolicy {
 public:
  SloAwarePolicy(const Options& options,
                 std::shared_ptr<InstanceMgr> instance_mgr);

  virtual ~SloAwarePolicy() = default;

  bool select_instances_pair(std::shared_ptr<Request> request) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(SloAwarePolicy);

  Options options_;
};

}  // namespace janus_service