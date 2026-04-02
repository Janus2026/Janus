#pragma once

#include "common/macros.h"
#include "loadbalance_policy.h"
#include "scheduler/managers/global_kvcache_mgr.h"

namespace janus_service {

class CacheAwareRouting final : public LoadBalancePolicy {
 public:
  CacheAwareRouting(std::shared_ptr<InstanceMgr> instance_mgr,
                    std::shared_ptr<GlobalKVCacheMgr> global_kvcache_mgr)
      : global_kvcache_mgr_(global_kvcache_mgr),
        LoadBalancePolicy(instance_mgr) {};

  virtual ~CacheAwareRouting() = default;

  bool select_instances_pair(std::shared_ptr<Request> request) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(CacheAwareRouting);

  void cost_function(
      const std::unordered_map<std::string, uint32_t>& overlap_scores,
      const uint32_t& max_block_num,
      const std::unordered_map<std::string, LoadMetrics>& load_metrics,
      const int64_t& max_waiting_requests_num,
      std::string* best_choice);

  std::shared_ptr<GlobalKVCacheMgr> global_kvcache_mgr_;
};

}  // namespace janus_service
