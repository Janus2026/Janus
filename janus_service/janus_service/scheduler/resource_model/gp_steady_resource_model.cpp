#include "gp_steady_resource_model.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>

namespace janus_service {

GPSteadyResourceModel::GPSteadyResourceModel(
    std::unique_ptr<GaussianProcess> gp_hbm,
    std::unique_ptr<GaussianProcess> gp_compute,
    std::unique_ptr<GaussianProcess> gp_bandwidth)
    : gp_hbm_(std::move(gp_hbm)),
      gp_compute_(std::move(gp_compute)),
      gp_bandwidth_(std::move(gp_bandwidth)) {}

ResourceNeeds GPSteadyResourceModel::calc_3d_resources(
    double token_rate, double avg_input_len, double avg_input_len2,
    double avg_output_len) const {
  // All 3 GPs use the same 3D input: [token_rate, avg_input_len, avg_output_len]
  Eigen::VectorXd x(3);
  x << token_rate, avg_input_len, avg_output_len;

  ResourceNeeds needs;
  needs.hbm_gb = std::max(gp_hbm_->predict_mean(x), 0.0) + 5.0;
  needs.compute_sm = std::clamp(gp_compute_->predict_mean(x), 0.0, 1.0);
  needs.bandwidth = std::clamp(gp_bandwidth_->predict_mean(x), 0.0, 1.0);

  LOG(INFO) << "@@ GP steady: input=(" << token_rate << ", " << avg_input_len
            << ", " << avg_input_len2 << ", " << avg_output_len
            << ") -> HBM=" << needs.hbm_gb
            << "GB, Compute=" << needs.compute_sm
            << ", Bandwidth=" << needs.bandwidth;
  return needs;
}

ResourceNeeds GPSteadyResourceModel::compute_resource_needs(
    int64_t model_heat) const {
  // Legacy fallback: convert heat (total tokens in window) to approximate
  // tokens/sec and use reasonable defaults for missing dimensions.
  static constexpr double kWindowSeconds = 15.0;
  static constexpr double kDefaultAvgInputLen = 100.0;
  static constexpr double kDefaultAvgOutputLen = 20.0;
  double token_rate = static_cast<double>(model_heat) / kWindowSeconds;
  double avg_input_len2 = kDefaultAvgInputLen * kDefaultAvgInputLen;
  return calc_3d_resources(token_rate, kDefaultAvgInputLen,
                           avg_input_len2, kDefaultAvgOutputLen);
}

int32_t GPSteadyResourceModel::compute_gpu_target(
    int64_t model_heat, const GpuHardwareSpec& hw) const {
  ResourceNeeds needs = compute_resource_needs(model_heat);

  // HBM needs normalization; compute_sm and bandwidth are already [0,1] per GPU
  double hbm_gpus = needs.hbm_gb / hw.hbm_per_gpu_gb;
  double compute_gpus = needs.compute_sm;
  double bandwidth_gpus = needs.bandwidth;

  int32_t target = static_cast<int32_t>(
      std::ceil(std::max({hbm_gpus, compute_gpus, bandwidth_gpus})));
  return std::max(target, 1);
}

std::string GPSteadyResourceModel::name() const { return "gp_steady"; }

}  // namespace janus_service
