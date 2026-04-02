#pragma once

#include <cstdint>
#include <string>

#include "common/types.h"

namespace janus_service {

class ResourceModel {
 public:
  virtual ~ResourceModel() = default;
  virtual int32_t compute_gpu_target(int64_t model_heat,
                                     const GpuHardwareSpec& hw) const = 0;
  virtual ResourceNeeds compute_resource_needs(int64_t model_heat) const = 0;
  virtual std::string name() const = 0;

  // GP steady pool interface: compute 3D resource needs from traffic stats.
  // Input: (token_rate, avg_input_len, avg_input_len², avg_output_len)
  // Default implementation delegates to compute_resource_needs(model_heat).
  virtual ResourceNeeds calc_3d_resources(double token_rate,
                                          double avg_input_len,
                                          double avg_input_len2,
                                          double avg_output_len) const {
    return compute_resource_needs(static_cast<int64_t>(token_rate));
  }

  // GP dynamic pool interface: compute required instance count for target SLO.
  // Default implementation delegates to compute_gpu_target(model_heat, hw).
  virtual int32_t calc_instance_number(double token_rate,
                                       double moment1,
                                       double moment2,
                                       double target_slo_rate) const {
    return compute_gpu_target(static_cast<int64_t>(token_rate),
                              GpuHardwareSpec{});
  }
};

}  // namespace janus_service
