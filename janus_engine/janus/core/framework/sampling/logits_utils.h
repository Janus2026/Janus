#pragma once
#include <torch/torch.h>

#include <memory>
#include <vector>

#if defined(USE_NPU)
#include "kernels/npu/janus_ops/janus_ops_api.h"
#endif
#include "kernels/ops_api.h"

namespace janus {

void apply_frequency_presence_penalties(
    torch::Tensor& logits,
    const torch::Tensor& unique_token_ids,
    const torch::Tensor& unique_token_counts,
    const torch::Tensor& frequency_penalties,
    const torch::Tensor& presence_penalties);

void apply_repetition_penalties(torch::Tensor& logits,
                                const torch::Tensor& unique_token_ids,
                                const torch::Tensor& penalties);

void apply_temperatures(torch::Tensor& logits,
                        const torch::Tensor& temperatures);

void apply_top_k_top_p(torch::Tensor& logits,
                       const torch::Tensor& temperatures,
                       const torch::Tensor& top_k,
                       const torch::Tensor& top_p);

}  // namespace janus
