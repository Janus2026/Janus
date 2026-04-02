#pragma once
#include <cstdint>
#include <utility>
#include <vector>

#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"

namespace janus {
namespace layer {

// Used to record information before padding
struct PaddingInfo {
  int64_t original_tokens = 0;  // Number of tokens before padding
  int64_t padded_tokens = 0;    // Number of tokens after padding
  bool active = false;          // Whether padding was performed
};

int64_t get_reduce_scatter_tokens(int64_t num_tokens,
                                  const ParallelArgs& parallel_args);

std::pair<torch::Tensor, PaddingInfo> pad_tokens(torch::Tensor x,
                                                 int64_t target_tokens);

// Prepare attention output for TP reduce scatter by applying the rank0
// residual add, padding to target_tokens, then scattering across TP ranks.
std::pair<torch::Tensor, PaddingInfo> reduce_scatter_attn_input(
    torch::Tensor x,
    const torch::Tensor& residual,
    int64_t target_tokens,
    const ParallelArgs& parallel_args);

int64_t get_dp_gather_tokens(const std::vector<int32_t>& dp_tokens,
                             const ParallelArgs& args);

torch::Tensor gather_global_tokens(const torch::Tensor& input,
                                   const std::vector<int32_t>& dp_tokens,
                                   const ParallelArgs& args);

// Unpadding logic after All Gather
// Simply slice out the original length
torch::Tensor unpad_tokens(torch::Tensor x, const PaddingInfo& pad_info);

// check if decoder should gather dp tokens before moe
bool need_dp_moe_gather(const ParallelArgs& args, bool enable_moe_all2all);

// gather tokens from all dp ranks before moe
torch::Tensor gather_dp_tokens(const torch::Tensor& input,
                               const ModelInputParams& params,
                               const ParallelArgs& args);

// given a tensor containing data from all DP ranks,
// returns a slice containing only the tokens for the current DP rank
torch::Tensor get_dp_local_slice(const torch::Tensor& input,
                                 const ModelInputParams& params,
                                 const ParallelArgs& args);

// check if all dp ranks are decode
bool all_dp_ranks_are_decode(const ModelInputParams& params);

}  // namespace layer
}  // namespace janus
