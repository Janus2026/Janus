#pragma once

#include <framework/distributed/process_group_cncl.hpp>

#include "process_group.h"

namespace janus {

constexpr int32_t local_device_count = 8;

class ProcessGroupImpl : public ProcessGroup {
 public:
  ProcessGroupImpl(int32_t global_rank,
                   int32_t world_size,
                   int32_t rank_size,
                   int32_t port,
                   bool trans,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device)
      : ProcessGroup(global_rank, world_size, device) {
    c10::intrusive_ptr<torch_mlu::ProcessGroupCNCL::Options> pg_options =
        torch_mlu::ProcessGroupCNCL::Options::create();
    pg_options->group_name = group_name;
    int32_t rank = global_rank;
    std::string local_host = host;
    if (world_size != rank_size) {
      auto [local_rank, group_ranks] =
          get_group_rank(world_size, global_rank, rank_size, trans);
      pg_options->global_ranks_in_group = group_ranks;
      rank = local_rank;
      int32_t server_ip_idx = group_ranks[0] / local_device_count;
      int32_t cur_ip_idx = global_rank / local_device_count;
      if (server_ip_idx == cur_ip_idx) {
        local_host = "127.0.0.1";
      }
    }

    auto store = create_tcp_store(local_host, port, rank);
    pg_ = std::make_unique<torch_mlu::ProcessGroupCNCL>(
        store, rank, rank_size, pg_options);
  }
};

}  // namespace janus