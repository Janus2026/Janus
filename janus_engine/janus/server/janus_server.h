#pragma once

#include <cstdint>

#include "api_service/api_service.h"
#include "core/distributed_runtime/collective_service.h"
#include "core/distributed_runtime/disagg_pd_service.h"
#include "core/distributed_runtime/pd_ooc_service.h"
#include "core/distributed_runtime/worker_service.h"
#include "core/framework/xtensor/xtensor_dist_service.h"

namespace janus {

class JanusServer final {
 public:
  JanusServer();
  ~JanusServer();

  bool start(std::unique_ptr<APIService> api_service);
  bool start(std::unique_ptr<DisaggPDService> disagg_pd_service,
             uint16_t disagg_pd_port);
  bool start(std::unique_ptr<PDOOCService> pd_ooc_service,
             uint16_t disagg_pd_port);
  bool start(std::shared_ptr<CollectiveService> service,
             const std::string& addr,
             const std::string& server_name);
  bool start(std::shared_ptr<WorkerService> service, const std::string& addr);
  bool start(std::shared_ptr<XTensorDistService> service,
             const std::string& addr);

  void run();
  void stop();

  bool has_initialized() const { return has_initialized_; }
  std::string listen_address() const { return listen_address_; }
  int listen_port() const { return listen_port_; }

 private:
  DISALLOW_COPY_AND_ASSIGN(JanusServer);
  bool create_server(google::protobuf::Service* service,
                     const std::string& addr,
                     int port,
                     const std::string& server_name);

 private:
  bool has_initialized_ = false;
  int listen_port_ = -1;
  std::string listen_address_;
  std::unique_ptr<brpc::Server> server_;
  std::unique_ptr<std::thread> running_thread_;
};

class JanusServerFactory {
 public:
  static std::unique_ptr<JanusServer> create_janus_server() {
    return std::make_unique<JanusServer>();
  }

 private:
  JanusServerFactory() = default;
  ~JanusServerFactory() = default;
  DISALLOW_COPY_AND_ASSIGN(JanusServerFactory);
};

}  // namespace janus
