#pragma once
#include <torch/torch.h>
#include <torch/types.h>

#include "constrained_decoding.h"
#include "util/threadpool.h"

namespace janus {

class RecConstrainedDecoding : public ConstrainedDecoding {
 public:
  RecConstrainedDecoding(uint64_t model_version,
                         const int32_t vocab_size,
                         torch::ScalarType dtype,
                         torch::Device device,
                         bool use_gen_threadpool_ = true);
  virtual ~RecConstrainedDecoding() = default;

  bool build_mask_cache() override;

  torch::Tensor generate_mask(
      const std::vector<std::vector<int32_t>>& generated_token_list) override;

 private:
  torch::Tensor generate_decode_mask(
      const std::vector<std::vector<int32_t>>& generated_token_list);

 private:
  constexpr static float PRE_MASK_FACTOR = -10000.0f;
  constexpr static int GEN_MASK_THREAD_NUM = 16;

 private:
  bool build_mask_cache_;
  bool use_gen_threadpool_;
  int32_t vocab_size_;
  uint64_t model_version_;
  torch::Device device_;
  torch::ScalarType dtype_;
  torch::Tensor first_token_mask_;
  std::unique_ptr<ThreadPool> gen_threadpool_;
};

}  // namespace janus
