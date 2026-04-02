// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/cutlass_extensions/common.cpp

#include "cutlass_extensions/common.hpp"

#include <glog/logging.h>

#include "core/platform/cuda/cuda_utils.h"

namespace janus {
namespace kernel {
namespace cuda {

int32_t get_sm_version_num() {
  int32_t major_capability, minor_capability;
  cudaDeviceGetAttribute(
      &major_capability, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(
      &minor_capability, cudaDevAttrComputeCapabilityMinor, 0);
  int32_t version_num = major_capability * 10 + minor_capability;
  return version_num;
}

int32_t get_device_sm_count(int device) {
  return janus::cuda::get_device_sm_count(device);
}

}  // namespace cuda
}  // namespace kernel
}  // namespace janus
