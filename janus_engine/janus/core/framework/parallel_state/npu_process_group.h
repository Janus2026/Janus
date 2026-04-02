#pragma once

#include <hccl/hccl_types.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "hccl/hccl.h"
#include "process_group.h"

namespace janus {

class ProcessGroupImpl : public ProcessGroup {
 public:
  // Constructor.
  ProcessGroupImpl(int rank,
                   int world_size,
                   const torch::Device& device,
                   HcclComm comm);

  ProcessGroupImpl(int rank,
                   int world_size,
                   int rank_size,
                   int port,
                   bool trans,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device);

  // Destructor.
  ~ProcessGroupImpl() override;

  void allreduce(torch::Tensor& input) override;

  void allgather(const torch::Tensor& input,
                 std::vector<torch::Tensor>& outputs) override;

 private:
  HcclComm comm_ = nullptr;
  c10_npu::NPUStream comm_stream_;
};

// TODO: LOG HcclGetErrorString(r)
#if defined(USE_NPU)
#define HCCLCHECK(cmd)                     \
  do {                                     \
    HcclResult r = cmd;                    \
    if (r != HCCL_SUCCESS) {               \
      LOG(FATAL) << "Failed, HCCL error."; \
    }                                      \
  } while (0)
#endif
}  // namespace janus