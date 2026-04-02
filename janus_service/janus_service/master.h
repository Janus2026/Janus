#pragma once

#include <brpc/server.h>

#include <thread>

#include "common/options.h"
#include "http_service/service.h"
#include "rpc_service/service.h"
#include "scheduler/scheduler.h"

namespace janus_service {

class Master {
 public:
  explicit Master(const Options& options);
  ~Master();

  bool start();
  void stop();

 private:
  bool start_http_server();
  bool start_rpc_server();

 private:
  Options options_;

  // Scheduler for scheduling requests and instances
  std::unique_ptr<Scheduler> scheduler_;

  // 1.For http service
  std::string http_server_address_;
  std::unique_ptr<janus_service::JanusHttpServiceImpl> http_service_;
  brpc::Server http_server_;
  std::unique_ptr<std::thread> http_server_thread_;

  // 2.For rpc service
  std::string rpc_server_address_;
  std::unique_ptr<janus_service::JanusRpcService> rpc_service_;
  brpc::Server rpc_server_;
  std::unique_ptr<std::thread> rpc_server_thread_;
};

}  // namespace janus_service
