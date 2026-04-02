#pragma once

#include <optional>
#include <string>
#include <vector>

#include "core/common/types.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "request_output.h"

namespace janus {

class LogprobState {
 public:
  LogprobState(int64_t num_prompt_tokens, size_t capacity);
  ~LogprobState() = default;

  // for generated tokens
  float get_acc_logprob(int64_t num_tokens);
  // Returns the accumulated logprob excluding the last generated token.
  float get_base_logprob(int64_t num_tokens);

  void generate_output_tokens_logprobs(
      size_t start_idx,
      size_t end_idx,
      const Tokenizer& tokenizer,
      std::optional<std::vector<LogProb>>& out_logprobs,
      bool skip_special_tokens,
      const std::vector<int32_t>& tokens);

  void update_logprob(size_t index, const Token& token, int64_t num_top_tokens);

  const std::vector<std::vector<float>>& get_top_logprobs() const {
    return top_logprobs_;
  }

  const std::vector<std::vector<int64_t>>& get_top_tokens() const {
    return top_tokens_;
  }

  const std::vector<std::optional<float>>& get_logprobs() const {
    return logprobs_;
  }

  void set_acc_logprob(float acc_logprob) { acc_logprob_ = acc_logprob; }

  void set_last_acc_token_idx(int64_t last_acc_token_idx) {
    last_acc_token_idx_ = last_acc_token_idx;
  }

 private:
  int64_t num_prompt_tokens_;
  std::vector<std::optional<float>> logprobs_;
  // accumulated log probability of the sequence
  float acc_logprob_ = 0.0;
  int64_t last_acc_token_idx_ = -1;
  // top k log probs
  std::vector<std::vector<int64_t>> top_tokens_;
  std::vector<std::vector<float>> top_logprobs_;
};

}  // namespace janus
