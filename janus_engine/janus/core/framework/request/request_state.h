#pragma once

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <vector>

#include "core/framework/sampling/sampling_params.h"
#include "mm_data.h"
#include "rec_type.h"
#include "request_output.h"
#include "sample_slot.h"
#include "stopping_checker.h"

namespace janus {

using OutputFunc = std::function<bool(const RequestOutput& output)>;
using OutputsFunc =
    std::function<std::vector<bool>(const std::vector<RequestOutput>& outputs)>;

class Call;

enum class RequestPriority { DEFAULT = 0, HIGH = 1, NORMAL = 2, LOW = 3 };

struct SchedulerParam {
  bool offline = false;
  int32_t ttlt_slo_ms = std::numeric_limits<int32_t>::max();
  int32_t ttft_slo_ms = std::numeric_limits<int32_t>::max();
  int32_t tpot_slo_ms = std::numeric_limits<int32_t>::max();
  int32_t tpot_priority_weight = 1;
  int32_t ttft_priority_weight = 1;
  int32_t ttlt_priority_weight = 1;
  int32_t priority_weight = 1;
  RequestPriority priority = RequestPriority::NORMAL;
};

struct RequestState final {
 public:
  RequestState() {}

  RequestState(const std::string& prompt,
               const std::vector<int32_t>& prompt_tokens,
               const RequestSamplingParam& sampling_param,
               const SchedulerParam& scheduler_param,
               const StoppingChecker& stopping_checker,
               size_t seq_capacity,
               size_t n,
               size_t best_of,
               bool logprobs,
               bool stream,
               bool echo,
               bool skip_special_tokens,
               bool enable_schedule_overlap,
               const OutputFunc& output_func,
               const OutputsFunc& outputs_func,
               const std::string& decode_address = "",
               std::optional<Call*> call = std::nullopt);

  RequestState(const std::string& prompt,
               const std::vector<int32_t>& prompt_tokens,
               torch::Tensor input_embedding,
               const RequestSamplingParam& sampling_param,
               const StoppingChecker& stopping_checker,
               size_t seq_capacity,
               size_t n,
               size_t best_of,
               bool logprobs,
               bool stream,
               bool echo,
               bool skip_special_tokens,
               bool enable_schedule_overlap,
               const OutputFunc& output_func,
               const OutputsFunc& outputs_func,
               const std::string& decode_address = "");

  RequestState(const std::string& prompt,
               const std::vector<int32_t>& prompt_tokens,
               const MMData& mm_data,
               const RequestSamplingParam& sampling_param,
               const StoppingChecker& stopping_checker,
               size_t seq_capacity,
               size_t n,
               size_t best_of,
               bool logprobs,
               bool stream,
               bool echo,
               bool skip_special_tokens,
               bool enable_schedule_overlap,
               const OutputFunc& output_func,
               const OutputsFunc& outputs_func,
               const std::string& decode_address = "");

  // for profiling run, only provide prompt tokens
  RequestState(const std::vector<int32_t>& prompt_tokens);

 public:
  // sampling parameters
  RequestSamplingParam sampling_param;

  // scheduling parameters
  SchedulerParam scheduler_param;

  // stopping criteria
  StoppingChecker stopping_checker;

  std::string prompt;

  std::vector<int32_t> prompt_tokens;

  bool stream = false;

  // max tokens for a seq
  size_t seq_capacity;

  size_t n;

  size_t best_of;

  bool echo = false;

  bool skip_special_tokens = true;

  OutputFunc output_func;

  // function to call when batch outputs is generated in disagg pd mode,
  // decode will send the batch outputs to prefill.
  OutputsFunc outputs_func;

  // decode address.
  std::string decode_address;
  std::string decode_rpc_address;

  // When true, prefill and decode happen on the same instance without
  // Mooncake transfer. KV blocks from prefill are reused directly for decode.
  bool local_pd = false;

  torch::Tensor input_embedding;

  // multimodal
  MMData mm_data;

  // whether to return log probabilities for output token.
  bool logprobs;

  bool enable_schedule_overlap = false;

  RecType rec_type = RecType::kNone;

  int32_t bos_token_id = 0;

  // The thread id of the thread pool in the response handler to ensure that
  // stream responses for the same request are executed sequentially during
  // multi-threaded stream processing.
  int32_t response_thread_id = -1;

  bool preempted = false;

  // This will be used in enable_scheduler_overlap
  bool handle_last_token_done = false;

  std::optional<Call*> call_;

  std::vector<SampleSlot> sample_slots;
};

}  // namespace janus
