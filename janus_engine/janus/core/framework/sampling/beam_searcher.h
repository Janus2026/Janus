#pragma once
#include <torch/torch.h>
#include <torch/types.h>

#if defined(USE_NPU)
#include "kernels/npu/janus_ops/janus_ops_api.h"
#endif

namespace janus {

struct BeamSearchOutput {
  torch::Tensor src_seq_idxes;  // [num_seq]
  torch::Tensor out_tokens;     // [num_seq]
  torch::Tensor out_logprobs;   // [num_seq]
};

class BeamSearcher {
 public:
  BeamSearcher() = default;

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  // logprobs: [num_seq]
  // top_tokens: [num_seq, top_k]
  // top_logprobs: [num_seq, top_k]
  BeamSearchOutput forward(const torch::Tensor& logprobs,
                           const torch::Tensor& top_tokens,
                           const torch::Tensor& top_logprobs) const;
};

}  // namespace janus