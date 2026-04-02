#pragma once
#include <torch/torch.h>

#include <memory>

#include "common/rec_model_utils.h"
#include "sampling_params.h"

namespace janus {

class Sampler;

class RecSampler {
 public:
  explicit RecSampler(RecPipelineType pipeline_type);
  ~RecSampler() = default;

  // logits: [batch_size, vocab_size]
  SampleOutput forward(torch::Tensor& logits,
                       const SamplingParameters& params) const;

 private:
  class SamplingStrategy {
   public:
    virtual ~SamplingStrategy() = default;
    virtual SampleOutput forward(torch::Tensor& logits,
                                 const SamplingParameters& params) const = 0;
  };

  class DefaultSamplingStrategy final : public SamplingStrategy {
   public:
    explicit DefaultSamplingStrategy(const Sampler& sampler);
    SampleOutput forward(torch::Tensor& logits,
                         const SamplingParameters& params) const override;

   private:
    const Sampler& sampler_;
  };

  class MultiRoundFastPathSamplingStrategy final : public SamplingStrategy {
   public:
    explicit MultiRoundFastPathSamplingStrategy(const Sampler& sampler);
    SampleOutput forward(torch::Tensor& logits,
                         const SamplingParameters& params) const override;

   private:
    const Sampler& sampler_;
  };

  static std::unique_ptr<SamplingStrategy> create_sampling_strategy(
      RecPipelineType type,
      const Sampler& sampler);

  std::unique_ptr<Sampler> sampler_;
  std::unique_ptr<SamplingStrategy> strategy_;
};

}  // namespace janus
