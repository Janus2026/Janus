#pragma once

#include <absl/container/flat_hash_set.h>

#include "api_service_impl.h"
#include "completion.pb.h"
#include "core/distributed_runtime/rec_master.h"
#include "rec.pb.h"
#include "stream_call.h"

namespace janus {

using CompletionCall =
    StreamCall<proto::CompletionRequest, proto::CompletionResponse>;

// a class to handle completion requests
class RecCompletionServiceImpl final : public APIServiceImpl<CompletionCall> {
 public:
  RecCompletionServiceImpl(RecMaster* master,
                           const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<CompletionCall> call);

 private:
  DISALLOW_COPY_AND_ASSIGN(RecCompletionServiceImpl);
  RecMaster* master_ = nullptr;
};

}  // namespace janus
