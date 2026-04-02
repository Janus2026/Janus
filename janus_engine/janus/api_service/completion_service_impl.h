#pragma once

#include <shared_mutex>
#include <unordered_map>

#include "api_service_impl.h"
#include "completion.pb.h"
#include "stream_call.h"

namespace janus {

using CompletionCall =
    StreamCall<proto::CompletionRequest, proto::CompletionResponse>;

// a class to handle completion requests
class CompletionServiceImpl final : public APIServiceImpl<CompletionCall> {
 public:
  CompletionServiceImpl(LLMMaster* master,
                        const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<CompletionCall> call);

  void process_async_rpc_impl(const proto::CompletionRequest* request);

  void add_model_master(const std::string& model, LLMMaster* master);
  void remove_model_master(const std::string& model);

 private:
  LLMMaster* get_model_master(const std::string& model) const;
  DISALLOW_COPY_AND_ASSIGN(CompletionServiceImpl);
  LLMMaster* master_ = nullptr;
  mutable std::shared_mutex llm_model_to_master_mutex_;
  std::unordered_map<std::string, LLMMaster*> llm_model_to_master_;
};

}  // namespace janus
