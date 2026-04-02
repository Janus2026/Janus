#include <brpc/server.h>
#include <google/protobuf/service.h>

#include "core/common/metrics.h"

namespace janus {

static void request_in_metric(void* context) {
  COUNTER_INC(server_request_in_total);
}

static void request_out_metric(void* context) {
  auto ctrl = reinterpret_cast<brpc::Controller*>(context);
  if (ctrl == nullptr) {
    LOG(ERROR) << "ctrl is nullptr";
    return;
  }

  if (!ctrl->Failed()) {
    COUNTER_INC(server_request_total_ok);
  } else {
    COUNTER_INC(server_request_total_fail);
    if (ctrl->ErrorCode() == brpc::ELIMIT) {
      COUNTER_INC(server_request_total_limit);
    }
  }
}

static void device_info_metric() {
  // TODO: get cpu device info
  GAUGE_SET(janus_cpu_num, 0);
  GAUGE_SET(janus_cpu_utilization, 0);

  // TODO: get gpu device info
  GAUGE_SET(janus_gpu_num, 0);
  GAUGE_SET(janus_gpu_utilization, 0);
}

}  // namespace janus
