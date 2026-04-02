#include "linear_resource_model.h"

#include <algorithm>
#include <cmath>

namespace janus_service {

LinearResourceModel::LinearResourceModel(double hbm_a, double hbm_b,
                                         double compute_a, double compute_b)
    : hbm_a_(hbm_a),
      hbm_b_(hbm_b),
      compute_a_(compute_a),
      compute_b_(compute_b) {}

int32_t LinearResourceModel::compute_gpu_target(
    int64_t model_heat, const GpuHardwareSpec& hw) const {
  double hbm_need = hbm_a_ * model_heat + hbm_b_;
  double compute_need = compute_a_ * model_heat + compute_b_;

  double hbm_gpus = hbm_need / hw.hbm_per_gpu_gb;
  double compute_gpus = compute_need / hw.compute_sm_per_gpu;

  int32_t target = static_cast<int32_t>(
      std::ceil(std::max(hbm_gpus, compute_gpus)));
  return std::max(target, 1);
}

std::string LinearResourceModel::name() const { return "linear"; }

ResourceNeeds LinearResourceModel::compute_resource_needs(
    int64_t model_heat) const {
  return {hbm_a_ * model_heat + hbm_b_, compute_a_ * model_heat + compute_b_, 0.0};
}

}  // namespace janus_service
