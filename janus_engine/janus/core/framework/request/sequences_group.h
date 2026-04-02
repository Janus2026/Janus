#pragma once

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "common.pb.h"
#include "core/framework/sampling/sampling_params.h"
#include "mm_data.h"
#include "sequence.h"
#include "stopping_checker.h"
#include "util/threadpool.h"

namespace janus {

class SequencesGroup {
 public:
  SequencesGroup(const std::string& prompt,
                 const std::vector<int32_t>& prompt_tokens,
                 const torch::Tensor& input_embedding,
                 const MMData& mm_data,
                 const SequenceParams& sequence_params);

  bool finished() const;

  bool expand_sequences(bool share_prefix);

  void generate_outputs(std::vector<SequenceOutput>& outputs,
                        const Tokenizer& tokenizer,
                        ThreadPool* thread_pool = nullptr);

  void process_beam_search();

  bool check_beam_search() {
    return sequence_params_.sampling_param->beam_width > 1;
  }

  std::vector<std::unique_ptr<Sequence>>& sequences() { return sequences_; }

  int32_t dp_rank() { return sequences_[0]->dp_rank(); }

  bool is_chunked_prefill_stage() const {
    return sequences_[0]->is_chunked_prefill_stage();
  }

  // mark all sequences as finished (used by rec model multi-round decoding)
  void finish();

 private:
  void add();

  void generate_outputs_parallel(std::vector<SequenceOutput>& outputs,
                                 const Tokenizer& tokenizer,
                                 ThreadPool* thread_pool = nullptr);

  // Generate output for multi-round beam search
  void generate_multi_round_output(std::vector<SequenceOutput>& outputs,
                                   const Tokenizer& tokenizer,
                                   const Sequence& base);

 private:
  const std::string& prompt_;                  // ref from request
  const std::vector<int32_t>& prompt_tokens_;  // ref from request
  const torch::Tensor& input_embedding_;       // ref from request
  const MMData& mm_data_;                      // ref from request
  SequenceParams sequence_params_;

 private:
  std::vector<std::unique_ptr<Sequence>> sequences_;
};

}  // namespace janus
