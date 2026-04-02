#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

#include "process_group.h"

namespace janus {

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
    c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> pg_options =
        c10d::ProcessGroupNCCL::Options::create();
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
    pg_options->group_name = group_name;
#endif
    int32_t rank = global_rank;
    if (world_size != rank_size) {
      auto [local_rank, group_ranks] =
          get_group_rank(world_size, global_rank, rank_size, trans);
      pg_options->global_ranks_in_group = group_ranks;
      rank = local_rank;
    }

    auto store = create_tcp_store(host, port, rank);
    pg_ = std::make_unique<c10d::ProcessGroupNCCL>(
        store, rank, rank_size, pg_options);
  }
};

}  // namespace janus
