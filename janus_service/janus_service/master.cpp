#include "master.h"

#include <csignal>

#include "common/global_gflags.h"
#include "common/types.h"
#include "common/utils.h"

namespace janus_service {

Master::Master(const Options& options) : options_(options) {
  scheduler_ = std::make_unique<Scheduler>(options);

  rpc_service_ =
      std::make_unique<janus_service::JanusRpcService>(options, scheduler_.get());

  http_service_ = std::make_unique<janus_service::JanusHttpServiceImpl>(
      options, scheduler_.get());
}

Master::~Master() { stop(); }

bool Master::start() {
  // 1. start http server
  http_server_thread_ =
      std::make_unique<std::thread>([this]() { start_http_server(); });

  // 2. start rpc server
  rpc_server_thread_ =
      std::make_unique<std::thread>([this]() { start_rpc_server(); });

  return true;
}

void Master::stop() {
  if (http_server_thread_ && http_server_thread_->joinable()) {
    http_server_thread_->join();
  }

  if (rpc_server_thread_ && rpc_server_thread_->joinable()) {
    rpc_server_thread_->join();
  }
}

bool Master::start_http_server() {
  if (http_server_.AddService(http_service_.get(),
                              brpc::SERVER_DOESNT_OWN_SERVICE,
                              // for testing
                              "/hello => Hello,"
                              "/v1/completions => Completions,"
                              "/v1/chat/completions => ChatCompletions,"
                              "/v1/embeddings => Embeddings,"
                              "/v1/models => Models,"
                              "/metrics => Metrics,"
                              "/model/triggers => ModelTriggers") != 0) {
    LOG(FATAL) << "Fail to add http service";
    return false;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = options_.http_idle_timeout_s();
  options.num_threads = options_.http_num_threads();
  options.max_concurrency = options_.http_max_concurrency();

  butil::EndPoint endpoint;
  if (!options_.server_host().empty()) {
    http_server_address_ =
        options_.server_host() + ":" + std::to_string(options_.http_port());
    if (butil::str2endpoint(http_server_address_.c_str(), &endpoint) < 0) {
      LOG(FATAL) << "Convert server_addr to endpoint failed: "
                 << http_server_address_;
      return false;
    }
  } else {
    endpoint = butil::EndPoint(butil::IP_ANY, options_.http_port());
  }

  if (http_server_.Start(endpoint, &options) != 0) {
    LOG(FATAL) << "Failed to start http server on: " << endpoint;
    return false;
  }

  LOG(INFO) << "Janus http server started on: " << endpoint;

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  http_server_.RunUntilAskedToQuit();
  return true;
}

bool Master::start_rpc_server() {
  if (rpc_server_.AddService(rpc_service_.get(),
                             brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(FATAL) << "Failed to add rpc service.";
    return false;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = options_.rpc_idle_timeout_s();
  options.num_threads = options_.rpc_num_threads();
  options.max_concurrency = options_.rpc_max_concurrency();

  butil::EndPoint endpoint;
  if (!options_.server_host().empty()) {
    rpc_server_address_ =
        options_.server_host() + ":" + std::to_string(options_.rpc_port());
    if (butil::str2endpoint(rpc_server_address_.c_str(), &endpoint) < 0) {
      LOG(FATAL) << "Convert server_addr to endpoint failed: "
                 << rpc_server_address_;
      return false;
    }
  } else {
    endpoint = butil::EndPoint(butil::IP_ANY, options_.rpc_port());
  }

  if (rpc_server_.Start(endpoint, &options) != 0) {
    LOG(FATAL) << "Failed to start rpc server on: " << endpoint;
    return false;
  }

  LOG(INFO) << "Janus rpc server started on: " << endpoint;

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  rpc_server_.RunUntilAskedToQuit();
  return true;
}

}  // namespace janus_service

static std::atomic<uint32_t> g_signal_received{0};
void shutdown_handler(int signal) {
  LOG(WARNING) << "Received signal " << signal << ", stopping master...";
  exit(1);
}

int main(int argc, char* argv[]) {
  // Initialize gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Initialize glog
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  LOG(INFO) << "Starting janus master service.";

  // check port available or not
  if (!janus_service::utils::is_port_available(FLAGS_http_server_port)) {
    LOG(ERROR)
        << "Http server port " << FLAGS_http_server_port
        << " is already in use. "
        << "Please specify a different port using --http_server_port flag.";
    return -1;
  }
  if (!janus_service::utils::is_port_available(FLAGS_rpc_server_port)) {
    LOG(ERROR)
        << "Rpc server port " << FLAGS_rpc_server_port << " is already in use. "
        << "Please specify a different port using --rpc_server_port flag.";
    return -1;
  }

  janus_service::Options options;
  options.server_host(FLAGS_server_host)
      .http_port(FLAGS_http_server_port)
      .http_idle_timeout_s(FLAGS_http_server_idle_timeout_s)
      .http_num_threads(FLAGS_http_server_num_threads)
      .http_max_concurrency(FLAGS_http_server_max_concurrency)
      .rpc_port(FLAGS_rpc_server_port)
      .rpc_idle_timeout_s(FLAGS_rpc_server_idle_timeout_s)
      .rpc_num_threads(FLAGS_rpc_server_num_threads)
      .rpc_max_concurrency(FLAGS_rpc_server_max_concurrency)
      .num_threads(FLAGS_num_threads)
      .max_concurrency(FLAGS_max_concurrency)
      .timeout_ms(FLAGS_timeout_ms)
      .etcd_addr(FLAGS_etcd_addr)
      .load_balance_policy(FLAGS_load_balance_policy)
      .murmur_hash3_seed(FLAGS_murmur_hash3_seed)
      .service_name(janus_service::utils::get_local_ip() + ":" +
                    std::to_string(FLAGS_rpc_server_port))
      .detect_disconnected_instance_interval(
          FLAGS_detect_disconnected_instance_interval)
      .enable_request_trace(FLAGS_enable_request_trace)
      .block_size(FLAGS_block_size)
      .tokenizer_path(FLAGS_tokenizer_path)
      .default_ttft_slo_ms(FLAGS_default_ttft_slo_ms)
      .lst_imh_pre_pull_ms(FLAGS_lst_imh_pre_pull_ms)
      .enable_mix_pd(FLAGS_enable_mix_pd)
      .slo_penalty_factor(FLAGS_slo_penalty_factor)
      .max_slo_expansions(FLAGS_max_slo_expansions)
      .disable_steady_pool(FLAGS_disable_steady_pool)
      .disable_elastic_pool(FLAGS_disable_elastic_pool)
      .elastic_instance_count(FLAGS_elastic_instance_count)
      .tensor_parallel_size(FLAGS_tensor_parallel_size);

  if (options.tensor_parallel_size() < 1) {
    LOG(ERROR) << "tensor_parallel_size must be >= 1, got "
               << options.tensor_parallel_size();
    return -1;
  }
  LOG(INFO) << "Tensor parallel size: " << options.tensor_parallel_size();

  if (options.disable_steady_pool() && options.disable_elastic_pool()) {
    LOG(ERROR) << "disable_steady_pool and disable_elastic_pool cannot both "
               << "be true.";
    return -1;
  }

  if (options.disable_steady_pool()) {
    LOG(INFO) << "Steady pool DISABLED: all models will use elastic pool "
              << "with PD disaggregation.";
    if (options.elastic_instance_count() == 1) {
      LOG(ERROR) << "elastic_instance_count must be 0 (all instances) or >= 2, "
                 << "got 1.";
      return -1;
    }
    if (options.elastic_instance_count() < 0) {
      LOG(ERROR) << "elastic_instance_count must be non-negative, got "
                 << options.elastic_instance_count();
      return -1;
    }
    if (options.elastic_instance_count() >= 2) {
      LOG(INFO) << "Elastic scaling DISABLED: fixed "
                << options.elastic_instance_count()
                << " instances per model.";
    } else {
      LOG(INFO) << "Elastic scaling DISABLED: using ALL available instances "
                << "per model.";
    }
  }

  if (options.disable_elastic_pool()) {
    LOG(INFO) << "Elastic pool DISABLED: all models will use steady pool only, "
              << "no PD disaggregation or dynamic scaling.";
  }

  janus_service::Master master(options);

  if (!master.start()) {
    LOG(ERROR) << "Failed to start master service.";
    return -1;
  }

  // install graceful shutdown handler
  (void)signal(SIGINT, shutdown_handler);
  (void)signal(SIGTERM, shutdown_handler);

  while (g_signal_received.load(std::memory_order_relaxed) == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // wait here
  master.stop();

  return 0;
}
