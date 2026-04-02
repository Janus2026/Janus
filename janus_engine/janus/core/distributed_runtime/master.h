#pragma once

#include <folly/Function.h>

#include <functional>
#include <future>
#include <vector>

#include "common/macros.h"
#include "common/options.h"
#include "common/rate_limiter.h"
#include "common/types.h"
#include "engine.h"
#include "framework/request/request_params.h"
namespace janus {

class Master {
 public:
  explicit Master(const Options& options, EngineType type);
  virtual ~Master() = default;
  virtual void run() = 0;
  virtual const Options& options() const { return options_; }

  virtual bool sleep() { return false; }

  virtual bool wakeup() { return false; }

  virtual bool wakeup(const WakeupOptions& options) { return false; }

  virtual bool link_d2d(const std::vector<std::string>& device_ips) {
    return false;
  }

  virtual bool unlink_d2d(const std::vector<std::string>& device_ips) {
    return false;
  }

  virtual bool resize(uint64_t new_kv_cache_pages) { return false; }

  MasterStatus get_master_status() const { return master_status_; }

  bool is_sleeping() const { return master_status_ != MasterStatus::WAKEUP; }

  void set_master_status(MasterStatus master_status) {
    master_status_ = master_status;
  }

  RateLimiter* get_rate_limiter() { return &rate_limiter_; }

 protected:
  Options options_;
  std::unique_ptr<Engine> engine_;
  RateLimiter rate_limiter_;
  MasterStatus master_status_{MasterStatus::WAKEUP};
};

std::unique_ptr<Master> create_master(const std::string& backend,
                                      const Options& options);

std::unique_ptr<Master> fork_master(Master* master, const Options& options);

}  // namespace janus
