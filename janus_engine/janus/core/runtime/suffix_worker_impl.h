#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "runtime/speculative_worker_impl.h"
#include "util/suffix_decoding_cache.h"

namespace janus {

// Suffix-based speculative decoding worker.
// Uses a suffix tree cache to generate draft tokens from previously seen
// patterns, without requiring a separate draft model.
class SuffixWorkerImpl : public SpeculativeWorkerImpl {
 public:
  SuffixWorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options);

  ~SuffixWorkerImpl() override = default;

 protected:
  std::optional<ForwardOutput> step_prefill(const ForwardInput& input) override;
  std::optional<ForwardOutput> step_decode(const ForwardInput& inputs) override;
  std::optional<ForwardOutput> step_empty(const ForwardInput& inputs) override;

 private:
  SampleOutput validate(const SamplingParameters& sampling_params,
                        const torch::Tensor& draft_token_ids,
                        const torch::Tensor& draft_probs,
                        const ForwardOutput& target_output);

 private:
  std::unique_ptr<SuffixDecodingCache> suffix_cache_;
  std::unordered_map<std::string, std::vector<int32_t>> suffix_recent_tokens_;
  std::unordered_set<std::string> suffix_active_decode_req_ids_;
};
}  // namespace janus
