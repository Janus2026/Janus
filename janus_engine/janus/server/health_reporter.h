#pragma once

#include <brpc/health_reporter.h>

#include "core/common/health_check_manager.h"

namespace janus {

// Custom health reporter for brpc server.
// Reports cluster health status based on worker connections.
class HealthReporter : public brpc::HealthReporter {
 public:
  static HealthReporter& instance() {
    static HealthReporter reporter;
    return reporter;
  }

  void GenerateReport(brpc::Controller* cntl,
                      google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);

    bool is_healthy = HealthCheckManager::instance().is_healthy();

    if (!is_healthy) {
      std::string reason = HealthCheckManager::instance().unhealthy_reason();
      LOG(ERROR) << "HealthReporter: cluster is unhealthy, reason: " << reason;
      cntl->http_response().set_status_code(503);  // Service Unavailable
    } else {
      cntl->http_response().set_status_code(200);  // OK
    }
  }
};

}  // namespace janus
