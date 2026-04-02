#pragma once
#include "comm_channel.h"
#include "runtime/forward_shared_memory_manager.h"
#include "runtime/options.h"

namespace janus {

class ShmChannel : public CommChannel {
 public:
  explicit ShmChannel(int dp_group,
                      int rank,
                      bool is_driver,
                      const runtime::Options& options);
  ~ShmChannel() = default;

  void execute_model_async(
      const RawForwardInput& input,
      folly::Promise<std::optional<RawForwardOutput>>& promise) override;

 private:
  bool execute_model_with_shm(const RawForwardInput& input,
                              RawForwardOutput& raw_output);

  bool enable_shm_ = false;
  std::unique_ptr<ForwardSharedMemoryManager> input_shm_manager_ = nullptr;
  std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager_ = nullptr;
};

}  // namespace janus
