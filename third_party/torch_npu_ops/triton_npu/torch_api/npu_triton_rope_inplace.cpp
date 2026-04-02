/* Copyright 2026 The Janus Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#include "operation_factory.h"
#include "triton_ops_api.h"

namespace janus::kernel::npu {

void rope_inplace(torch::Tensor& x,
                  torch::Tensor& sin,
                  torch::Tensor& cos,
                  uint32_t rope_dim) {
  auto org_shape = x.sizes();
  int32_t bsz = 0;
  int32_t head_num = 0;
  int32_t hidden_size = 0;
  if (x.dim() == 2) {
    bsz = x.size(0);
    head_num = 1;
    hidden_size = x.size(1);
  } else if (x.dim() == 3) {
    bsz = x.size(0);
    head_num = x.size(1);
    hidden_size = x.size(2);
    x = x.view({-1, hidden_size});
  }
  int32_t gridX = bsz * head_num;
  int32_t gridY = 1;
  int32_t gridZ = 1;
  int32_t x_stride = x.stride(0);
  int32_t rope_stride = cos.stride(0);
  auto npuStream = c10_npu::getCurrentNPUStream();
  rtStream_t stream = static_cast<rtStream_t>(npuStream.stream());

  auto& op = OperationFactory::instance().rope_inplace();
  auto ret = op.execute(stream, gridX, gridY, gridZ, [&](ArgsBuilder& ab) {
    ab.constructArgs(x.data_ptr(),
                     sin.data_ptr(),
                     cos.data_ptr(),
                     head_num,
                     x_stride,
                     rope_stride);
  });
  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for 'rope_inplace_kernel': " << ret;
  }

  if (org_shape.size() == 3) {
    x = x.view(org_shape);
  }
}

}  // namespace janus::kernel::npu
