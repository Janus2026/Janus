#pragma once
#include <cstdint>

#include "common/macros.h"
#include "common/types.h"
#include "util/threadpool.h"

namespace janus {

class BlockManager;
class Request;
class Sequence;
class Tokenizer;
class AsyncResponseProcessor final {
 public:
  AsyncResponseProcessor(const Tokenizer* tokenizer,
                         const std::optional<InstanceRole>& role,
                         bool enable_service_routing);
  virtual ~AsyncResponseProcessor() = default;

  void process_completed_request(std::shared_ptr<Request> request);

  void process_failed_request(std::shared_ptr<Request> request, Status status);

  // in disagg pd mode, decode send requests' responses to prefill
  void process_completed_requests(
      std::vector<std::shared_ptr<Request>>& requests);

  void process_stream_request(std::shared_ptr<Request> request);

  // in disagg pd mode, decode send requests' responses to prefill
  void process_stream_requests(std::vector<std::shared_ptr<Request>>& requests);

  void batch_process_stream_requests(
      std::vector<std::shared_ptr<Request>>& requests);

  void batch_process_completed_requests(
      std::vector<std::shared_ptr<Request>>& requests);
  // wait for all responses in queue to be handled
  void wait_completion();

 private:
  DISALLOW_COPY_AND_ASSIGN(AsyncResponseProcessor);
  // the threadpool to handle responses
  ThreadPool response_threadpool_;

  // the threadpool to handle rpc
  ThreadPool rpc_threadpool_;

  // the threadpool to generate outputs
  ThreadPool generate_output_threadpool_{16};

  // tokenizer instance to decode token ids
  std::unique_ptr<Tokenizer> tokenizer_;

  InstanceRole role_ = InstanceRole::DEFAULT;

  // for decode instance in disagg pd mode,
  // `True` means decode instance will response batch request_outputs to
  // prefill or janus service, this will decrease rpc cost.
  bool enable_batch_response_ = false;
};

}  // namespace janus
