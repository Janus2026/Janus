#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace janus {

class WorkerServer;

class SpawnWorkerServer final {
 public:
  explicit SpawnWorkerServer(const std::string& master_node_addr,
                             int local_rank,
                             int global_rank,
                             int world_size,
                             int device_idx,
                             int num_decoding_tokens,
                             int block_size,
                             bool enable_shm,
                             uint64_t input_shm_size,
                             uint64_t output_shm_size,
                             bool is_local,
                             bool enable_prefill_sp,
                             const std::string& task_type,
                             const std::string& worker_type,
                             const std::string& communication_backend);

  ~SpawnWorkerServer();

  void run();

  static void handle_signal(int signum);

 private:
  std::atomic<bool> done_{false};
  std::unique_ptr<WorkerServer> worker_server_;
};

}  // namespace janus
