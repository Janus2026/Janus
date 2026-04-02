#pragma once

#include "common/macros.h"
#include "resource_model.h"

namespace janus_service {

class LinearResourceModel final : public ResourceModel {
 public:
  LinearResourceModel(double hbm_a, double hbm_b,
                      double compute_a, double compute_b);
  ~LinearResourceModel() override = default;

  int32_t compute_gpu_target(int64_t model_heat,
                             const GpuHardwareSpec& hw) const override;
  ResourceNeeds compute_resource_needs(int64_t model_heat) const override;
  std::string name() const override;

 private:
  DISALLOW_COPY_AND_ASSIGN(LinearResourceModel);
  double hbm_a_, hbm_b_;
  double compute_a_, compute_b_;
};

}  // namespace janus_service
