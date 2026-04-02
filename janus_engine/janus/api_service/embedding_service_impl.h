#pragma once
#include <absl/container/flat_hash_set.h>

#include "api_service/api_service_impl.h"
#include "api_service/call.h"
#include "api_service/non_stream_call.h"
#include "core/distributed_runtime/vlm_master.h"
#include "embedding.pb.h"

namespace janus {

using EmbeddingCall =
    NonStreamCall<proto::EmbeddingRequest, proto::EmbeddingResponse>;

// a class to handle completion requests
class EmbeddingServiceImpl final : public APIServiceImpl<EmbeddingCall> {
 public:
  EmbeddingServiceImpl(LLMMaster* master,
                       const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<EmbeddingCall> call);

 private:
  DISALLOW_COPY_AND_ASSIGN(EmbeddingServiceImpl);
  LLMMaster* master_ = nullptr;
};

using MMEmbeddingCall =
    NonStreamCall<proto::MMEmbeddingRequest, proto::EmbeddingResponse>;
class MMEmbeddingServiceImpl : public APIServiceImpl<MMEmbeddingCall> {
 public:
  MMEmbeddingServiceImpl(VLMMaster* master,
                         const std::vector<std::string>& models);
  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<MMEmbeddingCall> call);

 private:
  DISALLOW_COPY_AND_ASSIGN(MMEmbeddingServiceImpl);
  VLMMaster* master_ = nullptr;
};

}  // namespace janus
