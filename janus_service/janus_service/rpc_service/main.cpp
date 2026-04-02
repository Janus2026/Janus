#include <brpc/server.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/global_gflags.h"
#include "common/types.h"
#include "common/utils.h"
#include "rpc_service/service.h"

int main(int argc, char* argv[]) {
  // Initialize gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Initialize glog
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Dump all gflags: " << std::endl
            << google::CommandlineFlagsIntoString();
  google::FlushLogFiles(google::INFO);

  LOG(INFO) << "Starting janus rpc service, port: " << FLAGS_port;

  if (!janus_service::utils::is_port_available(FLAGS_port)) {
    LOG(ERROR) << "Port " << FLAGS_port << " is already in use. "
               << "Please specify a different port using --port flag.";
    return 1;
  }

  // create janus service
  janus_service::Options service_options;
  janus_service::JanusRpcService service(service_options, nullptr);

  // Initialize brpc server
  std::string server_address = "0.0.0.0:" + std::to_string(FLAGS_port);
  brpc::Server server;
  if (server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(ERROR) << "Failed to add service to server";
    return -1;
  }

  butil::EndPoint endpoint;
  if (!FLAGS_listen_addr.empty()) {
    if (butil::str2endpoint(FLAGS_listen_addr.c_str(), &endpoint) < 0) {
      LOG(ERROR) << "Invalid listen address:" << FLAGS_listen_addr;
      return -1;
    }
  } else {
    endpoint = butil::EndPoint(butil::IP_ANY, FLAGS_port);
  }

  // Start the server.
  brpc::ServerOptions options;
  options.idle_timeout_sec = FLAGS_idle_timeout_s;
  options.num_threads = FLAGS_num_threads;
  options.max_concurrency = FLAGS_max_concurrency;
  options.idle_timeout_sec = FLAGS_idle_timeout_s;
  if (server.Start(endpoint, &options) != 0) {
    LOG(ERROR) << "Fail to start Brpc rpc server";
    return -1;
  }

  LOG(INFO) << "Janus rpc service listening on " << server_address;

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  server.RunUntilAskedToQuit();

  return 0;
}
