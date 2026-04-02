#include "rpc_service/client.h"

#include <glog/logging.h>

namespace janus_service {

// magic number, TODO: move to config file or env var
static constexpr int kHeartbeatInterval = 3;  // in seconds

JanusRpcClient::JanusRpcClient(const std::string& instace_name,
                             const std::string& master_addr,
                             const ChannelOptions& options)
    : instance_name_(instace_name), master_addr_(master_addr) {
  brpc::ChannelOptions chan_options;
  chan_options.protocol = options.protocol;
  chan_options.connection_type = options.connection_type;
  chan_options.timeout_ms = options.timeout_ms /*milliseconds*/;
  chan_options.max_retry = options.max_retry;
  if (master_channel_.Init(master_addr_.c_str(),
                           options.load_balancer.c_str(),
                           &chan_options) != 0) {
    LOG(ERROR) << "Fail to initialize brpc channel to server " << master_addr_;
    return;
  }
  master_stub_ = std::make_unique<proto::JanusRpcService_Stub>(&master_channel_);

  // heartbeat thread
  heartbeat_thread_ =
      std::make_unique<std::thread>(&JanusRpcClient::heartbeat, this);
}

JanusRpcClient::~JanusRpcClient() {
  exited_ = true;
  if (heartbeat_thread_) {
    heartbeat_thread_->join();
  }
}

// TODO: send metainfo/metrics to master ?
void JanusRpcClient::heartbeat() {
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(kHeartbeatInterval));
    if (!register_inst_done_) continue;

    brpc::Controller cntl;
    proto::HeartbeatRequest req;
    req.set_name(instance_name_);
    // TODO: set req.cache_event and req.load_metrics
    proto::Status res;
    master_stub_->Heartbeat(&cntl, &req, &res, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << instance_name_
                 << " failed to send heartbeat to master: " << cntl.ErrorText();
      ;
    } else if (!res.ok()) {
      LOG(ERROR) << instance_name_
                 << " failed to send heartbeat to master, status: " << res.ok();
    }
  }
}

ErrorCode JanusRpcClient::register_instance() {
  InstanceMetaInfo metainfo;
  metainfo.name = instance_name_;
  return register_instance(metainfo);
}

ErrorCode JanusRpcClient::register_instance(const InstanceMetaInfo& metainfo) {
  brpc::Controller cntl;
  proto::InstanceMetaInfo req;
  req.set_name(metainfo.name);
  if (metainfo.type == InstanceType::PREFILL) {
    req.set_type(proto::InstanceType::PREFILL);
  } else if (metainfo.type == InstanceType::DECODE) {
    req.set_type(proto::InstanceType::DECODE);
  } else if (metainfo.type == InstanceType::MIX) {
    req.set_type(proto::InstanceType::MIX);
  } else {
    req.set_type(proto::InstanceType::DEFAULT);
  }
  proto::StatusCode res;
  master_stub_->RegisterInstance(&cntl, &req, &res, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << instance_name_
               << " failed to send register_instance to master: "
               << cntl.ErrorText();
    ;
  } else if (res.status_code() != ConvertErrorCode::to_int(ErrorCode::OK)) {
    LOG(ERROR) << instance_name_
               << " failed to send register_instance to master: "
               << "res = " << res.status_code();
  } else {
    // register instance success
    register_inst_done_ = true;
  }
  return ConvertErrorCode::from_int(res.status_code());
}

}  // namespace janus_service
