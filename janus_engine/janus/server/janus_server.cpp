#include "janus_server.h"

#include <brpc/server.h>
#include <butil/at_exit.h>

#include "core/common/global_flags.h"
#include "health_reporter.h"

namespace janus {

JanusServer::JanusServer() { butil::AtExitManager exit_manager; }

JanusServer::~JanusServer() {
  stop();

  if (running_thread_ && running_thread_->joinable()) {
    running_thread_->join();
  }
}

bool JanusServer::start(std::unique_ptr<APIService> service) {
  const int kMaxRetries = 5;
  const int kRetryIntervalSec = 1;

  for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
    server_ = std::make_unique<brpc::Server>();
    if (FLAGS_node_rank == 0) {
      if (server_->AddService(service.get(),
                              brpc::SERVER_DOESNT_OWN_SERVICE,
                              "v1/completions => CompletionsHttp,"
                              "v1/sample => SampleHttp,"
                              "v1/chat/completions => ChatCompletionsHttp,"
                              "v1/embeddings => EmbeddingsHttp,"
                              "v1/models => ModelsHttp,"
                              "v1/image/generation => ImageGenerationHttp,"
                              "v1/rerank => RerankHttp,"
                              "v1/messages => AnthropicMessagesHttp,"
                              "v2/repository/index => ModelVersionsHttp,"
                              "get_free_port => GetFreePortHttp,"
                              "fork_master => ForkMasterHttp,"
                              "sleep => SleepHttp,"
                              "wakeup => WakeupHttp,"
                              "link_d2d => LinkD2DHttp,"
                              "unlink_d2d => UnlinkD2DHttp,"
                              "resize => ResizeHttp") != 0) {
        LOG(ERROR) << "Fail to add api service";
        return false;
      }
    } else if (FLAGS_enable_xtensor) {
      if (server_->AddService(service.get(),
                              brpc::SERVER_DOESNT_OWN_SERVICE,
                              "get_free_port => GetFreePortHttp,"
                              "fork_master => ForkMasterHttp") != 0) {
        LOG(ERROR) << "Fail to add api service";
        return false;
      }
    }

    brpc::ServerOptions options;
    options.idle_timeout_sec = FLAGS_rpc_idle_timeout_s;
    options.num_threads = FLAGS_num_threads;
    options.health_reporter = &HealthReporter::instance();
    if (server_->Start(FLAGS_port, &options) == 0) {
      LOG(INFO) << "Brpc Server started on port " << FLAGS_port
                << ", idle_timeout_s: " << FLAGS_rpc_idle_timeout_s
                << ", num_threads: " << FLAGS_num_threads;

      listen_address_ =
          std::string(butil::endpoint2str(server_->listen_address()).c_str());
      listen_port_ = FLAGS_port;
      has_initialized_ = true;
      server_->RunUntilAskedToQuit();
      return true;
    }

    if (attempt < kMaxRetries - 1) {
      LOG(WARNING) << "Port " << FLAGS_port << " unavailable, retrying in "
                   << kRetryIntervalSec << "s (" << (attempt + 1) << "/"
                   << kMaxRetries << ")";
      sleep(kRetryIntervalSec);
    }
  }

  LOG(ERROR) << "Failed to start server on port " << FLAGS_port << " after "
             << kMaxRetries << " attempts";
  return false;
}

bool JanusServer::start(std::unique_ptr<DisaggPDService> service,
                       uint16_t /*disagg_pd_port*/) {
  std::string addr;
  if (!FLAGS_host.empty()) {
    addr = FLAGS_host + ":0";
  }
  if (!create_server(
          (google::protobuf::Service*)(service.get()), addr, 0, "Disagg PD")) {
    return false;
  }

  has_initialized_ = true;
  server_->Join();
  return true;
}

bool JanusServer::start(std::unique_ptr<PDOOCService> service,
                       uint16_t /*disagg_pd_port*/) {
  std::string addr;
  if (!FLAGS_host.empty()) {
    addr = FLAGS_host + ":0";
  }
  if (!create_server(
          (google::protobuf::Service*)(service.get()), addr, 0, "PD OOC")) {
    return false;
  }

  has_initialized_ = true;
  server_->Join();
  return true;
}

bool JanusServer::start(std::shared_ptr<CollectiveService> service,
                       const std::string& addr,
                       const std::string& server_name) {
  // Port 0 = OS-assigned: retries get a new port each time, so retrying helps.
  // Fixed port: retrying the same port is pointless (conflict won't self-heal).
  const bool is_ephemeral = (addr.size() > 2 && addr.substr(addr.size() - 2) == ":0");
  const int kMaxRetries = is_ephemeral ? 3 : 1;
  const int kRetryIntervalSec = 1;

  for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
    if (create_server((google::protobuf::Service*)(service.get()),
                      addr,
                      -1,
                      server_name)) {
      running_thread_ =
          std::make_unique<std::thread>([this, service = std::move(service)]() {
            has_initialized_ = true;
            server_->Join();
          });
      return true;
    }
    if (attempt < kMaxRetries - 1) {
      LOG(WARNING) << server_name << " failed to bind " << addr
                   << ", retrying " << (attempt + 1) << "/" << kMaxRetries;
      sleep(kRetryIntervalSec);
    }
  }
  LOG(ERROR) << server_name << " failed to bind " << addr << " after "
             << kMaxRetries << " attempts";
  return false;
}

bool JanusServer::start(std::shared_ptr<WorkerService> service,
                       const std::string& addr) {
  server_ = std::make_unique<brpc::Server>();
  if (server_->AddService(service.get(), brpc::SERVER_DOESNT_OWN_SERVICE) !=
      0) {
    LOG(ERROR) << "Fail to add DistributeWorker service";
    return false;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = FLAGS_rpc_idle_timeout_s;
  options.num_threads = FLAGS_num_threads;
  listen_address_ = addr;
  if (server_->Start(addr.c_str(), &options) != 0) {
    LOG(ERROR) << "Failed to start distribute server on address: " << addr;
    return false;
  }
  listen_port_ = server_->listen_address().port;
  LOG(INFO) << "DistributeWorker started on address "
            << server_->listen_address()
            << ", idle_timeout_sec: " << FLAGS_rpc_idle_timeout_s
            << ", num_threads: " << FLAGS_num_threads;

  return true;
}

bool JanusServer::start(std::shared_ptr<XTensorDistService> service,
                       const std::string& addr) {
  server_ = std::make_unique<brpc::Server>();
  if (server_->AddService(service.get(), brpc::SERVER_DOESNT_OWN_SERVICE) !=
      0) {
    LOG(ERROR) << "Fail to add XTensorDist service";
    return false;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = FLAGS_rpc_idle_timeout_s;
  options.num_threads = FLAGS_num_threads;
  listen_address_ = addr;
  if (server_->Start(addr.c_str(), &options) != 0) {
    LOG(ERROR) << "Failed to start XTensorDist server on address: " << addr;
    return false;
  }
  listen_port_ = server_->listen_address().port;
  LOG(INFO) << "XTensorDist server started on address "
            << server_->listen_address()
            << ", idle_timeout_sec: " << FLAGS_rpc_idle_timeout_s
            << ", num_threads: " << FLAGS_num_threads;

  return true;
}

bool JanusServer::create_server(google::protobuf::Service* service,
                               const std::string& addr,
                               int port,
                               const std::string& server_name) {
  server_ = std::make_unique<brpc::Server>();
  if (server_->AddService(service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(ERROR) << "Fail to add " << server_name << " service";
    return false;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = FLAGS_rpc_idle_timeout_s;
  options.num_threads = FLAGS_num_threads;
  butil::EndPoint endpoint;
  if (!addr.empty()) {
    if (butil::str2endpoint(addr.c_str(), &endpoint) < 0) {
      LOG(FATAL) << "Convert address to endpoint failed: " << addr;
      return false;
    }
  } else {
    endpoint = butil::EndPoint(butil::IP_ANY, port);
  }

  if (server_->Start(endpoint, &options) != 0) {
    LOG(ERROR) << "Failed to start " << server_name
               << " server on address: " << endpoint;
    return false;
  }
  listen_address_ =
      std::string(butil::endpoint2str(server_->listen_address()).c_str());
  listen_port_ = server_->listen_address().port;
  LOG(INFO) << server_name << " server started on address "
            << server_->listen_address()
            << ", idle_timeout_sec: " << FLAGS_rpc_idle_timeout_s
            << ", num_threads: " << FLAGS_num_threads;

  return true;
}

void JanusServer::run() {
  if (has_initialized_) {
    return;
  }

  has_initialized_ = true;
  server_->Join();
}

void JanusServer::stop() {
  server_->Stop(0);
  server_->Join();
}

}  // namespace janus
