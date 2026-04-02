#include "slo_aware_policy.h"

#include "common/global_gflags.h"

namespace janus_service {

SloAwarePolicy::SloAwarePolicy(const Options& options,
                               std::shared_ptr<InstanceMgr> instance_mgr)
    : options_(options), LoadBalancePolicy(instance_mgr) {}

bool SloAwarePolicy::select_instances_pair(std::shared_ptr<Request> request) {
  if (request->token_ids.empty()) {
    return instance_mgr_->get_next_instance_pair(request->model, &request->routing);
  }

  // select instances pair based on slo
  if (!instance_mgr_->select_instance_pair_on_slo(request)) {
    LOG(ERROR) << "Select instances based on the SLO failed!";
    return false;
  }

  return true;
}

}  // namespace janus_service