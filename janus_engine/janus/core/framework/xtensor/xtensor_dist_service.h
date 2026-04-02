#pragma once

#include <torch/torch.h>

#include "common/macros.h"
#include "util/threadpool.h"
#include "xtensor_dist.pb.h"

namespace janus {

class XTensorDistService : public proto::XTensorDist {
 public:
  XTensorDistService(int32_t global_rank,
                     int32_t world_size,
                     const torch::Device& device);
  ~XTensorDistService() = default;

  // Mark service as initialized
  void set_initialized(bool initialized) { initialized_ = initialized; }

  // Service functions
  void Hello(::google::protobuf::RpcController* controller,
             const proto::Status* request,
             proto::Status* response,
             ::google::protobuf::Closure* done) override;

  // Memory info query (get available/total memory before init)
  void GetMemoryInfo(::google::protobuf::RpcController* controller,
                     const proto::Status* request,
                     proto::MemoryInfoResponse* response,
                     ::google::protobuf::Closure* done) override;

  // Initialize PhyPagePool with specified number of pages
  void InitPhyPagePool(::google::protobuf::RpcController* controller,
                       const proto::InitPhyPagePoolRequest* request,
                       proto::Status* response,
                       ::google::protobuf::Closure* done) override;

  // KV tensor operations (partial mapping by offsets)
  void MapToKvTensors(::google::protobuf::RpcController* controller,
                      const proto::KvTensorRequest* request,
                      proto::Status* response,
                      ::google::protobuf::Closure* done) override;

  void UnmapFromKvTensors(::google::protobuf::RpcController* controller,
                          const proto::KvTensorRequest* request,
                          proto::Status* response,
                          ::google::protobuf::Closure* done) override;

  // Weight pages allocation from GlobalXTensor
  void AllocWeightPages(::google::protobuf::RpcController* controller,
                        const proto::AllocWeightPagesRequest* request,
                        proto::Status* response,
                        ::google::protobuf::Closure* done) override;

  void FreeWeightPages(::google::protobuf::RpcController* controller,
                       const proto::FreeWeightPagesRequest* request,
                       proto::Status* response,
                       ::google::protobuf::Closure* done) override;

  // Get XTensor offsets for KV cache blocks (used in PD disaggregation)
  void GetXTensorOffsets(::google::protobuf::RpcController* controller,
                         const proto::GetXTensorOffsetsRequest* request,
                         proto::GetXTensorOffsetsResponse* response,
                         ::google::protobuf::Closure* done) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(XTensorDistService);

 private:
  bool initialized_;
  int32_t global_rank_;
  int32_t world_size_;
  torch::Device device_;
  ThreadPool threadpool_{4};
};

}  // namespace janus
