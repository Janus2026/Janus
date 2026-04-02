#pragma once

#include <mutex>
#include <unordered_map>

#include "common/call_data.h"
#include "common/threadpool.h"
#include "common/janus/output.h"
#include "common/janus/status.h"

namespace janus_service {

class ResponseHandler final {
 public:
  ResponseHandler() = default;
  ~ResponseHandler() = default;

  bool send_delta_to_client(std::shared_ptr<ChatCallData> call_data,
                            std::unordered_set<size_t>* first_message_sent,
                            bool include_usage,
                            const std::string& request_id,
                            int64_t created_time,
                            const std::string& model,
                            const llm::RequestOutput& output);
  bool send_result_to_client(std::shared_ptr<ChatCallData> call_data,
                             const std::string& request_id,
                             int64_t created_time,
                             const std::string& model,
                             const llm::RequestOutput& req_output);

  bool send_delta_to_client(std::shared_ptr<CompletionCallData> call_data,
                            bool include_usage,
                            const std::string& request_id,
                            int64_t created_time,
                            const std::string& model,
                            const llm::RequestOutput& output);
  bool send_result_to_client(std::shared_ptr<CompletionCallData> call_data,
                             const std::string& request_id,
                             int64_t created_time,
                             const std::string& model,
                             const llm::RequestOutput& req_output);
};

}  // namespace janus_service
