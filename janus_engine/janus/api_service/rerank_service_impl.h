#pragma once
#include <absl/container/flat_hash_set.h>

#include <atomic>
#include <functional>
#include <vector>

#include "api_service/api_service_impl.h"
#include "api_service/call.h"
#include "api_service/non_stream_call.h"
#include "rerank.pb.h"

namespace janus {

using RerankCall = NonStreamCall<proto::RerankRequest, proto::RerankResponse>;

struct RerankRequestOutput {
  int32_t index = 0;
  std::string document = "";
  float score = 0.0f;

  RerankRequestOutput(int32_t index, std::string document, float score)
      : index(index), document(std::move(document)), score(score) {}
};

// Score computer function type: computes scores from request outputs
// Returns vector of RerankRequestOutput with computed scores
using ScoreComputer = std::function<std::vector<RerankRequestOutput>(
    const std::vector<std::string>& documents,
    const std::vector<RequestOutput>& req_outputs)>;

// Shared context for async aggregation of rerank sub-request results
// Template parameter allows different score computation strategies
struct RerankContext {
  std::shared_ptr<RerankCall> call;
  std::vector<std::string> documents;
  std::string model;
  std::string request_id;
  int32_t top_n;

  std::vector<RequestOutput> req_outputs;
  std::atomic<size_t> pending_count;

  ScoreComputer compute_scores;

  RerankContext(std::shared_ptr<RerankCall> call,
                std::vector<std::string> documents,
                std::string model,
                std::string request_id,
                int32_t top_n,
                size_t num_requests,
                ScoreComputer compute_scores)
      : call(std::move(call)),
        documents(std::move(documents)),
        model(std::move(model)),
        request_id(std::move(request_id)),
        top_n(top_n),
        pending_count(num_requests),
        compute_scores(std::move(compute_scores)) {
    req_outputs.resize(num_requests);
  }

  void on_complete(size_t index, RequestOutput output) {
    req_outputs[index] = std::move(output);

    if (pending_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      finalize();
    }
  }

  void finalize();
};

class RerankServiceImpl : public APIServiceImpl<RerankCall> {
 public:
  RerankServiceImpl(LLMMaster* master, const std::vector<std::string>& models);

  virtual void process_async_impl(std::shared_ptr<RerankCall> call);

 protected:
  DISALLOW_COPY_AND_ASSIGN(RerankServiceImpl);
  LLMMaster* master_ = nullptr;
};

}  // namespace janus
