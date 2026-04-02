#pragma once

#include <torch/torch.h>

#include <cmath>
#include <vector>

#include "dit_cache_impl.h"

namespace janus {

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

class TaylorSeer : public DitCacheImpl {
 public:
  TaylorSeer() = default;
  ~TaylorSeer() = default;

  TaylorSeer(const TaylorSeer&) = delete;
  TaylorSeer& operator=(const TaylorSeer&) = delete;
  TaylorSeer(TaylorSeer&&) = default;
  TaylorSeer& operator=(TaylorSeer&&) = default;

  void init(const DiTCacheConfig& cfg) override;

  // Reset all cached derivatives and internal state
  void reset_cache();

  // Mark the beginning of a new inference step
  void mark_step_begin();

  // Compute the approximate value for the current step
  torch::Tensor approximate_value();

  // Update internal caches with the new observation Y
  void update(const torch::Tensor& Y);

  bool on_before_block(const CacheBlockIn& blockin) override;
  CacheBlockOut on_after_block(const CacheBlockIn& blockin) override;

  bool on_before_step(const CacheStepIn& stepin) override;
  CacheStepOut on_after_step(const CacheStepIn& stepin) override;

 private:
  // Compute approximate derivatives of Y using previous steps
  std::pair<std::vector<torch::Tensor>, std::vector<bool>>
  approximate_derivative(const torch::Tensor& Y);

 private:
  bool use_cache_ = false;
  int n_derivatives_;
  int order_;
  int skip_interval_steps_;
  int last_non_approximated_step_;

  std::vector<torch::Tensor> dY_prev_;
  std::vector<torch::Tensor> dY_current_;
  std::vector<bool> valid_prev_;
  std::vector<bool> valid_current_;
};

}  // namespace janus
