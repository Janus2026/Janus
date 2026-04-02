#pragma once
#include <string>
#include <unordered_map>

#include "dit_cache_config.h"
#include "dit_cache_type.h"

namespace janus {

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

class DitCacheImpl {
 public:
  DitCacheImpl() = default;
  virtual ~DitCacheImpl() = default;

  virtual void init(const DiTCacheConfig& cfg) = 0;

  virtual bool on_before_block(const CacheBlockIn& blockin) = 0;
  virtual CacheBlockOut on_after_block(const CacheBlockIn& blockin) = 0;

  virtual bool on_before_step(const CacheStepIn& stepin) = 0;
  virtual CacheStepOut on_after_step(const CacheStepIn& stepin) = 0;

 protected:
  int64_t num_inference_steps_;
  int64_t warmup_steps_;
  int64_t current_step_;
  TensorMap buffers;

  static torch::Tensor get_tensor_or_empty(const TensorMap& m,
                                           const std::string& k);
  static bool is_similar(const torch::Tensor& lhs,
                         const torch::Tensor& rhs,
                         float threshold);
};

std::unique_ptr<DitCacheImpl> create_dit_cache(const DiTCacheConfig& cfg);

}  // namespace janus
