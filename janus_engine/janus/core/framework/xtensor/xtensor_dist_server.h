#pragma once

#include <brpc/server.h>
#include <torch/torch.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "common/macros.h"
#include "options.h"
#include "xtensor_dist.pb.h"

namespace janus {

class XTensorDistServer {
 public:
  XTensorDistServer(int local_rank,
                    const std::string& master_node_addr,
                    std::atomic<bool>& done,
                    const torch::Device& device,
                    const xtensor::Options& options);
  ~XTensorDistServer();

 private:
  DISALLOW_COPY_AND_ASSIGN(XTensorDistServer);

  void create_server(const xtensor::Options& options,
                     std::atomic<bool>& done,
                     const std::string& master_node_addr,
                     const torch::Device& device,
                     int world_size,
                     int global_rank,
                     int local_rank);

  bool sync_master_node(const std::string& master_node_addr,
                        proto::AddressInfo& addr_info,
                        proto::CommUniqueIdList& uids);

 private:
  std::unique_ptr<std::thread> server_thread_;
  std::string server_name_;
};

}  // namespace janus
