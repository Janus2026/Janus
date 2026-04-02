#pragma once

#include <brpc/channel.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/etcd_client.h"
#include "distributed_runtime/engine.h"
#include "forward_params.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/request_output.h"
#include "scheduler/scheduler.h"
#include "xservice.pb.h"
namespace janus {

class XServiceClient {
 public:
  static XServiceClient* get_instance() {
    static XServiceClient xservice_client;
    return &xservice_client;
  }

  ~XServiceClient();
  bool init(const std::string& etcd_addr,
            const std::string& instance_name = "",
            const BlockManagerPool* block_manager_pool = nullptr);
  void set_scheduler(Scheduler* scheduler);
  void set_engine(Engine* engine);
  bool initialize_done() { return initialize_done_; }

  std::string get_instance_name();
  void register_instance(const InstanceInfo& instance_info);
  void heartbeat();
  InstanceInfo get_instance_info(const std::string& instance_name);
  std::vector<std::string> get_static_decode_list();
  std::vector<std::string> get_static_prefill_list();

  // get all janus_service addrs
  std::vector<std::string> get_all_xservice_addrs();

  // response generation tokens to janus service
  std::vector<bool> generations(const std::vector<RequestOutput>& outputs);

 private:
  bool register_instance_with_retry(const std::string& key,
                                    const std::string& value);
  bool reconcile_registration();
  void reconcile_registration_loop();

  void handle_master_service_watch(const etcd::Response& response);
  void handle_xservices_watch(const etcd::Response& response);

  // connect to specific janus_service
  bool connect_to_xservice(const std::string& xservice_addr);

  // remove the connection to specific janus_service
  void disconnect_xservice(const std::string& xservice_addr);

  // call rpc with current master stub atomically.
  bool with_master_stub(
      const std::function<void(janus_service::proto::JanusRpcService_Stub*)>& fn,
      std::string* master_addr);

  // find stub by address, caller should hold mutex_
  janus_service::proto::JanusRpcService_Stub* find_stub_locked(
      const std::string& xservice_addr);

 private:
  XServiceClient() = default;

  std::atomic_bool exited_{false};
  std::atomic_bool register_done_{false};
  bool initialize_done_ = false;
  std::string instance_name_;
  std::string incarnation_id_;
  std::string registration_key_;
  std::string registration_value_;

  std::string master_xservice_addr_;
  std::unordered_map<std::string, std::unique_ptr<brpc::Channel>>
      xservice_channels_;
  std::unordered_map<std::string,
                     std::unique_ptr<janus_service::proto::JanusRpcService_Stub>>
      xservice_stubs_;
  std::unique_ptr<std::thread> heartbeat_thread_;
  std::unique_ptr<std::thread> reconcile_thread_;

  std::shared_mutex mutex_;
  std::mutex registration_mutex_;
  brpc::ChannelOptions chan_options_;
  std::unique_ptr<EtcdClient> etcd_client_;
  const BlockManagerPool* block_manager_pool_ = nullptr;  // not own
  Scheduler* scheduler_ = nullptr;                        // not own
  Engine* engine_ = nullptr;  // not own, for xtensor info
};

}  // namespace janus
