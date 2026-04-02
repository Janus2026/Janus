#pragma once

#include <memory>
#include <string>

#include "parallel_args.h"
#include "process_group.h"

namespace janus {

class CollectiveCommunicator {
 public:
  CollectiveCommunicator(int global_rank,
                         int world_size,
                         int dp_size,
                         int ep_size);
  ~CollectiveCommunicator() = default;

  void create_process_groups(const std::string& master_addr,
                             const torch::Device& device);

  // init communicator and return parallel args.
  const ParallelArgs* parallel_args();

 private:
  std::unique_ptr<ParallelArgs> parallel_args_;
  std::unique_ptr<ProcessGroup> process_group_;
  std::unique_ptr<ProcessGroup> dp_local_process_group_;
  std::unique_ptr<ProcessGroup> tp_group_;
  std::unique_ptr<ProcessGroup> single_rank_group_;
  // Reserved owner for a future standalone SP group. The current code path
  // aliases ParallelArgs::sp_group_ to tp_group_ instead of constructing a
  // separate communicator, so this stays empty for now.
  std::unique_ptr<ProcessGroup> sp_group_;
  std::unique_ptr<ProcessGroup> moe_tp_group_;
  std::unique_ptr<ProcessGroup> moe_ep_group_;
};

}  // namespace janus
