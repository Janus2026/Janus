#pragma once

#include "linear.h"

namespace janus {
namespace layer {

class LmHead : public torch::nn::ModuleHolder<RowParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<RowParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = RowParallelLinearImpl;

  LmHead(const ModelContext& context)
      : ModuleHolder(std::make_shared<RowParallelLinearImpl>(
            // NOTE: Quantization should NOT be used for the final language
            // modeling head (lm_head). The output logits must remain in high
            // precision (typically bfloat16/float16) for numerical stability
            // and correct evaluation of loss and predictions. Always use
            // unquantized weights here.
            context.get_model_args().hidden_size(),
            context.get_model_args().vocab_size(),
            /*bias=*/false,
            /*input_is_parallelized=*/false,
            /*enable_result_reduction=*/true,
            QuantArgs{},  // do not use quantization for lm_head!
            context.get_parallel_args().tp_group_,
            context.get_tensor_options())) {}
};

}  // namespace layer
}  // namespace janus
