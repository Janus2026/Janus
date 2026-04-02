#pragma once

#include <torch/torch.h>

#include <optional>

namespace janus {
struct ModelOutput {
  // [num_tokens, hidden_size]
  torch::Tensor hidden_states;
  // [num_tokens, hidden_size]
  torch::Tensor residual;
  // [num_tokens, ...]
  torch::Tensor aux_hidden_states;

  ModelOutput() = default;

  // Constructor with only hidden_states (for backward compatibility)
  explicit ModelOutput(const torch::Tensor& hidden_states)
      : hidden_states(hidden_states) {}

  explicit ModelOutput(const torch::Tensor& hidden_states,
                       const torch::Tensor& residual)
      : hidden_states(hidden_states), residual(residual) {}

  // Constructor with optional residual for multi-device compatibility
  explicit ModelOutput(const torch::Tensor& hidden_states,
                       const std::optional<torch::Tensor>& residual)
      : hidden_states(hidden_states) {
    if (residual.has_value()) {
      this->residual = residual.value();
    }
  }

  explicit ModelOutput(
      std::pair<torch::Tensor, torch::Tensor> hidden_states_and_residual)
      : hidden_states(hidden_states_and_residual.first),
        residual(hidden_states_and_residual.second) {}

  ModelOutput(torch::Tensor hidden_states,
              torch::Tensor residual,
              torch::Tensor aux_hidden_states)
      : hidden_states(hidden_states),
        residual(residual),
        aux_hidden_states(aux_hidden_states) {}
};
}  // namespace janus