#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mooncake_transfer_engine.h"

namespace janus {

// Weight transfer using Mooncake transfer engine for remote weight loading
class MooncakeWeightTransfer {
 public:
  MooncakeWeightTransfer(int16_t listen_port, const torch::Device& device);
  ~MooncakeWeightTransfer() = default;

  bool initialize();

  bool register_global_xtensor();

  bool link_d2d(const std::string& remote_addr);
  bool link_d2d(const std::vector<std::string>& remote_addrs);
  bool unlink_d2d(const std::string& remote_addr);
  bool unlink_d2d(const std::vector<std::string>& remote_addrs);

  bool pull_weights(const std::string& remote_addr,
                    uint64_t src_offset,
                    uint64_t dst_offset,
                    size_t size);

  bool push_weights(const std::string& remote_addr,
                    uint64_t src_offset,
                    uint64_t dst_offset,
                    size_t size);

 private:
  int16_t listen_port_;
  int32_t device_id_;
  std::string addr_;
  uint64_t cluster_id_ = 0;
  std::unique_ptr<MooncakeTransferEngine> mooncake_te_;
  bool initialized_ = false;
};

}  // namespace janus
