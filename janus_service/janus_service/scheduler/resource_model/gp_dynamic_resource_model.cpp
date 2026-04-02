#include "gp_dynamic_resource_model.h"

#include <algorithm>
#include <cmath>
#include <glog/logging.h>

namespace janus_service {

GPDynamicResourceModel::GPDynamicResourceModel(
    std::unique_ptr<GaussianProcess> gp_slo,
    int32_t max_instance_count)
    : gp_slo_(std::move(gp_slo)),
      max_instance_count_(max_instance_count) {}

double GPDynamicResourceModel::predict_slo_rate(
    double token_rate, double moment1, double moment2,
    int32_t instance_count) const {
  Eigen::VectorXd x(4);
  x << token_rate, moment1, moment2, static_cast<double>(instance_count);
  return std::clamp(gp_slo_->predict_mean(x), 0.0, 1.0);
}

int32_t GPDynamicResourceModel::calc_instance_number(
    double token_rate, double moment1, double moment2,
    double target_slo_rate) const {
  // Binary search: find minimum instance_count in [1, max_instance_count_]
  // such that predicted SLO rate >= target_slo_rate.
  // Assumes SLO rate is monotonically non-decreasing with instance_count.
  int32_t lo = 1, hi = max_instance_count_;
  int32_t result = max_instance_count_;

  while (lo <= hi) {
    int32_t mid = lo + (hi - lo) / 2;
    double slo = predict_slo_rate(token_rate, moment1, moment2, mid);

    if (slo >= target_slo_rate) {
      result = mid;
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }

  return result;
}

int32_t GPDynamicResourceModel::compute_gpu_target(
    int64_t model_heat, const GpuHardwareSpec& /*hw*/) const {
  // Default SLO target 0.95 for legacy interface
  return calc_instance_number(static_cast<double>(model_heat), 0.0, 0.0, 0.95);
}

ResourceNeeds GPDynamicResourceModel::compute_resource_needs(
    int64_t /*model_heat*/) const {
  // Dynamic model's primary output is instance count, not resource decomposition
  return ResourceNeeds{};
}

std::string GPDynamicResourceModel::name() const { return "gp_dynamic"; }

}  // namespace janus_service
