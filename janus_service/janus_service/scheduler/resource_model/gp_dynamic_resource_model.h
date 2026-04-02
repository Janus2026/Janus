#pragma once

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "gaussian_process.h"
#include "resource_model.h"

namespace janus_service {

// GP-based resource model for the dynamic (elastic) pool.
// Uses a single GP to predict SLO satisfaction rate:
//   GP: 4D input -> SLO satisfaction rate [0, 1]
//
// Input dimensions:
//   (token_rate, prompt_length_moment1, prompt_length_moment2, instance_count)
//
// To determine the minimum instance count for a target SLO rate,
// binary search is performed over the instance_count dimension.
class GPDynamicResourceModel final : public ResourceModel {
 public:
  GPDynamicResourceModel(std::unique_ptr<GaussianProcess> gp_slo,
                          int32_t max_instance_count = 64);
  ~GPDynamicResourceModel() override = default;

  // Legacy interface.
  int32_t compute_gpu_target(int64_t model_heat,
                             const GpuHardwareSpec& hw) const override;
  ResourceNeeds compute_resource_needs(int64_t model_heat) const override;
  std::string name() const override;

  // Primary GP interface: binary search for minimum instance count
  // that achieves the target SLO satisfaction rate.
  int32_t calc_instance_number(double token_rate,
                               double moment1,
                               double moment2,
                               double target_slo_rate) const override;

  // Predict SLO satisfaction rate for given traffic and instance count.
  double predict_slo_rate(double token_rate,
                          double moment1,
                          double moment2,
                          int32_t instance_count) const;

 private:
  DISALLOW_COPY_AND_ASSIGN(GPDynamicResourceModel);

  std::unique_ptr<GaussianProcess> gp_slo_;
  int32_t max_instance_count_;
};

}  // namespace janus_service
