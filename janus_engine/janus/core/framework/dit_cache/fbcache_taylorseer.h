#pragma once

#include <torch/torch.h>

#include "dit_cache_impl.h"
#include "taylorseer.h"

namespace janus {

class FBCacheTaylorSeer : public DitCacheImpl {
 public:
  FBCacheTaylorSeer() = default;
  ~FBCacheTaylorSeer() override = default;

  FBCacheTaylorSeer(const FBCacheTaylorSeer&) = delete;
  FBCacheTaylorSeer& operator=(const FBCacheTaylorSeer&) = delete;
  FBCacheTaylorSeer(FBCacheTaylorSeer&&) = default;
  FBCacheTaylorSeer& operator=(FBCacheTaylorSeer&&) = default;

  void init(const DiTCacheConfig& cfg) override;

  bool on_before_block(const CacheBlockIn& blockin) override;
  CacheBlockOut on_after_block(const CacheBlockIn& blockin) override;

  bool on_before_step(const CacheStepIn& stepin) override;
  CacheStepOut on_after_step(const CacheStepIn& stepin) override;

 private:
  std::pair<torch::Tensor, torch::Tensor> apply_prev_hidden_states_residual(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states);

  bool can_use_cache(const torch::Tensor& first_hidden_states_residual);

 private:
  std::unique_ptr<TaylorSeer> taylorseer;
  float residual_diff_threshold_;
  bool use_cache_;
};

}  // namespace janus
