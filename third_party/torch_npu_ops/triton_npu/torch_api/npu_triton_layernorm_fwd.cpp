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

constexpr int64_t MAX_CORES = 65535;
constexpr int64_t MAX_FUSED_BYTES = 65536;

inline int64_t next_power_of_2(int64_t n) {
  if (n <= 1) return 1;
  uint64_t val = static_cast<uint64_t>(n - 1);
  return 1LL << (64 - __builtin_clzll(val ? val : 1));
}

torch::Tensor layer_norm_fwd(torch::Tensor& x,
                             torch::Tensor& weight,
                             torch::Tensor& bias,
                             double eps,
                             const std::optional<torch::Tensor>& z,
                             int64_t group_size,
                             bool norm_before_gate,
                             bool is_rms_norm) {
  // TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  // TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");

  c10::IntArrayRef x_shape_og = x.sizes();
  int64_t last_dim = x.size(-1);
  torch::Tensor x_2d = x.reshape({-1, last_dim});

  // TORCH_CHECK(x_2d.stride(-1) == 1, "x stride(-1) must be 1");
  // TORCH_CHECK(x_2d.dim() == 2, "x must be 2-dimensional (M, N)");

  const auto M = x_2d.size(0);
  const auto N = x_2d.size(1);

  const int64_t group_size_val = group_size;
  // TORCH_CHECK(N % group_size_val == 0, "N must be divisible by group_size");
  const int64_t ngroups = N / group_size_val;

  torch::Tensor z_2d;
  if (z.has_value()) {
    z_2d = z->reshape({-1, last_dim});
  }

  // TORCH_CHECK(weight.dim() == 1 && weight.size(0) == N,
  //            "weight must be 1-dimensional with size N");
  // TORCH_CHECK(weight.stride(-1) == 1, "weight stride(-1) must be 1");

  torch::Tensor out_tensor = torch::empty_like(x_2d);
  torch::Tensor mean, rstd;
  if (!is_rms_norm) {
    mean = torch::empty({ngroups * M},
                        torch::dtype(torch::kFloat32).device(x.device()));
  }
  rstd = torch::empty({ngroups * M},
                      torch::dtype(torch::kFloat32).device(x.device()));

  const int64_t elem_size = x.element_size();
  const int64_t MAX_FUSED_SIZE = MAX_FUSED_BYTES / elem_size;
  const int64_t BLOCK_N =
      std::min(MAX_FUSED_SIZE, next_power_of_2(group_size_val));

  const int64_t warp_base = BLOCK_N / 256;
  const int64_t num_warps = std::clamp<int64_t>(warp_base, 1, 8);

  auto npuStream = c10_npu::getCurrentNPUStream();
  rtStream_t stream = static_cast<rtStream_t>(npuStream.stream());

  int32_t gridCoreNum = std::min(M, MAX_CORES);
  int32_t gridNgroups = ngroups;
  int32_t gridZ = 1;

  void* x_2dPtr = x_2d.data_ptr();
  void* out_tensorPtr = out_tensor.data_ptr();
  void* weightPtr = weight.data_ptr();
  void* biasPtr = nullptr;
  if (bias.defined()) {
    biasPtr = bias.data_ptr();
  }
  void* z_2dPtr = nullptr;
  if (z_2d.defined()) {
    z_2dPtr = z_2d.data_ptr();
  }
  void* meanPtr = nullptr;
  if (mean.defined()) {
    meanPtr = mean.data_ptr();
  }
  void* rstdPtr = rstd.data_ptr();
  int32_t stride_x_row = x_2d.stride(0);
  int32_t stride_y_row = out_tensor.stride(0);
  int32_t stride_z_row = 0;
  if (z.has_value()) {
    stride_z_row = z_2d.stride(0);
  }

  auto& op = OperationFactory::instance().layer_norm_fwd();

  auto ret =
      op.execute(stream, gridCoreNum, gridNgroups, gridZ, [&](ArgsBuilder& ab) {
        ab.constructArgs(x_2dPtr,
                         out_tensorPtr,
                         weightPtr,
                         z_2dPtr,
                         // meanPtr, in qwen3-next this input won't be needed.
                         rstdPtr,
                         stride_x_row,
                         stride_y_row,
                         stride_z_row,
                         static_cast<int32_t>(M),
                         static_cast<int32_t>(group_size),
                         static_cast<float>(eps));
      });
  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for 'layer_norm_fwd_kernel': " << ret;
  }
  return out_tensor.reshape(x_shape_og);
}
}  // namespace janus::kernel::npu
