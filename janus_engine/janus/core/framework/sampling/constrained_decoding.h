#pragma once
#include <c10/core/TensorOptions.h>
#include <torch/torch.h>
#include <torch/types.h>

namespace janus {

// Constrained decoding is used to ensure that the generated content
// conforms to specific formats or rules.
class ConstrainedDecoding {
 public:
  virtual ~ConstrainedDecoding() = default;

  // Precompute and cache fixed constraint masks (e.g., static vocabulary
  // whitelists) to avoid redundant calculations during token generation.
  // Returns: true if cache built successfully, false otherwise
  virtual bool build_mask_cache() = 0;

  // Generate dynamic constraint mask based on already generated token
  // sequences. This mask will be applied to filter invalid tokens.
  //
  // Input: generated_token_list - 2D vector of token IDs, where each inner
  // vector represents the generated tokens for a single sequence in the batch
  // (format:[sequence_num][token_ids])
  // Output: tensor of shape [sequence_num, vocab_size], where 0.0f
  // indicates allowed tokens and a large negative number indicates forbidden
  // tokens for each sequence, the usage is to filter invalid tokens by adding
  // the mask to the model logits.
  virtual torch::Tensor generate_mask(
      const std::vector<std::vector<int32_t>>& generated_token_list) = 0;
};
}  // namespace janus
