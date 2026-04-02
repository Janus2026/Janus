#include <brpc/server.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include "common/global_gflags.h"
#include "common/options.h"
#include "http_service/service.h"

int main(int argc, char** argv) {
  // Initialize gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Initialize glog
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  LOG(INFO) << "Starting janus http service, port: " << FLAGS_port;

  janus_service::Options service_options;
  janus_service::JanusHttpServiceImpl service_impl(service_options, nullptr);

  // register http methods here
  brpc::Server server;
  if (server.AddService(&service_impl,
                        brpc::SERVER_DOESNT_OWN_SERVICE,
                        "/hello => Hello,"
                        "/v1/completions => Completions,") != 0) {
    LOG(ERROR) << "Fail to add brpc http service";
    return false;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = FLAGS_idle_timeout_s;
  options.num_threads = FLAGS_num_threads;
  options.max_concurrency = FLAGS_max_concurrency;
  if (server.Start(FLAGS_port, &options) != 0) {
    LOG(ERROR) << "Failed to start brpc http server on port " << FLAGS_port;
    return false;
  }

  LOG(INFO) << "Janus http server started on port " << FLAGS_port
            << ", idle_timeout_sec: " << FLAGS_idle_timeout_s
            << ", num_threads: " << FLAGS_num_threads
            << ", max_concurrency: " << FLAGS_max_concurrency;

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  server.RunUntilAskedToQuit();

  return 0;
}
