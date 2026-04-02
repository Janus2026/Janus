// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/scaled_mm_c3x_sm120.cu

#include "c3x/scaled_mm_helper.hpp"
#include "c3x/scaled_mm_kernels.hpp"

// Quantized GEMM operations using CUTLASS 3.x API for sm120 (Blackwell).

#if defined ENABLE_SCALED_MM_SM120 && ENABLE_SCALED_MM_SM120

namespace janus::kernel::cuda {

void cutlass_scaled_mm_sm120(torch::Tensor& c,
                             torch::Tensor const& a,
                             torch::Tensor const& b,
                             torch::Tensor const& a_scales,
                             torch::Tensor const& b_scales,
                             std::optional<torch::Tensor> const& bias) {
  dispatch_scaled_mm(c,
                     a,
                     b,
                     a_scales,
                     b_scales,
                     bias,
                     cutlass_scaled_mm_sm120_fp8,
                     nullptr,   // int8 not supported on SM120
                     nullptr);  // blockwise not supported
}

}  // namespace janus::kernel::cuda

#endif
