#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <nlohmann/json.hpp>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include "acl/acl.h"
#include "aclnn_replace_token.h"
#include "core/common/macros.h"
#include "core/kernels/npu/utils.h"
#include "janus_ops_api.h"

namespace janus::kernel::npu {

// Used by schedule overlap on NPU.
// This wrapper prepares the next decode-step input under the ACLNN update
// rule.
// Inputs:
//   src: sampled tokens from the previous step.
// Outputs:
//   dst: current-step input token tensor, updated in place after replacement.
void replace_token(torch::Tensor& dst, torch::Tensor& src) {
  check_tensor(dst, "dst", "replace_token");
  check_tensor(src, "src", "replace_token");
  aclTensor* dst_ids = nullptr;
  aclTensor* src_ids = nullptr;
  int32_t device_id = dst.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  create_acltensor(&dst_ids, dst);
  create_acltensor(&src_ids, src);
  uint64_t workspace_size = 0;
  aclOpExecutor* executor;
  CHECK_ACL_SUCCESS(aclnnReplaceTokenGetWorkspaceSize(
                        dst_ids, src_ids, dst_ids, &workspace_size, &executor),
                    "replace_token: failed to get workspace size");
  void* workspace_addr = nullptr;
  CHECK_ACL_SUCCESS(
      aclnnReplaceToken(workspace_addr, workspace_size, executor, stream),
      "replace_token: failed to replace token");
  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                    "replace_token: failed to synchronize stream");
  aclDestroyTensor(dst_ids);
  aclDestroyTensor(src_ids);
}
}  // namespace janus::kernel::npu
