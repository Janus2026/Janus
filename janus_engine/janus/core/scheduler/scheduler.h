#pragma once

#include <absl/time/time.h>

#include <memory>
#include <string>

#include "framework/request/request.h"

namespace janus {

class SchedulerBase {
 public:
  virtual ~SchedulerBase() = default;

  // scheduler forward execute
  virtual void step(const absl::Duration& timeout) = 0;

  // offline running
  virtual void generate() = 0;

  // incr/decr pending requests
  virtual void incr_pending_requests(size_t count) {}
  virtual void decr_pending_requests() {}
  virtual size_t num_pending_requests() { return 0; }
};

class Scheduler : public SchedulerBase {
 public:
  virtual ~Scheduler() = default;

  // add a new request to scheduler.
  virtual bool add_request(std::shared_ptr<Request>& request) = 0;

  virtual uint32_t get_waiting_requests_num() const = 0;

  virtual void get_latency_metrics(std::vector<int64_t>& ttft,
                                   std::vector<int64_t>& tbt) = 0;

  virtual const InstanceInfo& get_instance_info() = 0;

  virtual void profile_ttft() {}
  virtual void profile_tpot() {}
  virtual void post_register_link() {}
};

}  // namespace janus
