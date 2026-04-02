#pragma once

#include <torch/torch.h>

#include <optional>
#include <string>

namespace janus {
struct ModelInputParams;

namespace layer {

struct AttentionMetadata;

// Builder class for AttentionMetadata to avoid circular dependency.
// This class handles building AttentionMetadata from ModelInputParams,
// allowing attention_metadata.h to not depend on model_input_params.h.
class AttentionMetadataBuilder {
 public:
  // Build AttentionMetadata from ModelInputParams with default compute_dtype
  // ("float").
  static AttentionMetadata build(
      const ModelInputParams& params,
      const std::optional<torch::Tensor>& attn_mask = {});

  // Build AttentionMetadata from ModelInputParams with specified compute_dtype.
  static AttentionMetadata build(
      const ModelInputParams& params,
      const std::string& compute_dtype,
      const std::optional<torch::Tensor>& attn_mask = {});
};

}  // namespace layer
}  // namespace janus
