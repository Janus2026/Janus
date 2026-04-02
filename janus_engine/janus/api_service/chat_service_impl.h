#pragma once

#include <shared_mutex>
#include <unordered_map>

#include "api_service/api_service_impl.h"
#include "api_service/stream_call.h"
#include "chat.pb.h"
#include "multimodal.pb.h"

namespace janus {

class RecMaster;

using ChatCall = StreamCall<proto::ChatRequest, proto::ChatResponse>;

// a class to handle completion requests
class ChatServiceImpl final : public APIServiceImpl<ChatCall> {
 public:
  // Constructor for LLM backend
  ChatServiceImpl(LLMMaster* master, const std::vector<std::string>& models);

  // Constructor for Rec backend (LlmRec only, e.g., Qwen3)
  ChatServiceImpl(RecMaster* master, const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<ChatCall> call);

  void process_async_rpc_impl(const proto::ChatRequest* request);

  void add_model_master(const std::string& model, LLMMaster* master);
  void remove_model_master(const std::string& model);

 private:
  void process_rec_chat_request(std::shared_ptr<ChatCall> call);
  LLMMaster* get_model_master(const std::string& model) const;

  DISALLOW_COPY_AND_ASSIGN(ChatServiceImpl);

  LLMMaster* master_ = nullptr;
  RecMaster* rec_master_ = nullptr;
  mutable std::shared_mutex llm_model_to_master_mutex_;
  std::unordered_map<std::string, LLMMaster*> llm_model_to_master_;
  const std::string tool_call_parser_format_;
  const std::string reasoning_parser_format_;
};

class VLMMaster;
using MMChatCall = StreamCall<proto::MMChatRequest, proto::ChatResponse>;

// a class to handle mm chat completion requests
class MMChatServiceImpl : public APIServiceImpl<MMChatCall> {
 public:
  MMChatServiceImpl(VLMMaster* master, const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<MMChatCall> call);

 private:
  DISALLOW_COPY_AND_ASSIGN(MMChatServiceImpl);

  VLMMaster* master_ = nullptr;
};

}  // namespace janus
