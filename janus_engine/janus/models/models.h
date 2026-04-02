#pragma once

// USE_NPU_TORCH: Temporary flag used for debugging qwen3 torch NPU graph
// capture. This variable may be removed in the future.
#if defined(USE_NPU) && defined(USE_NPU_TORCH)
#include "llm/qwen3.h"  // IWYU pragma: keep
#include "llm/qwen3_next.h"
#elif defined(USE_NPU)
#include "dit/pipeline_flux.h"                // IWYU pragma: keep
#include "dit/pipeline_flux_control.h"        // IWYU pragma: keep
#include "dit/pipeline_flux_fill.h"           // IWYU pragma: keep
#include "llm/npu/deepseek_mtp.h"             // IWYU pragma: keep
#include "llm/npu/deepseek_v2.h"              // IWYU pragma: keep
#include "llm/npu/deepseek_v3.h"              // IWYU pragma: keep
#include "llm/npu/deepseek_v32.h"             // IWYU pragma: keep
#include "llm/npu/deepseek_v32_mtp.h"         // IWYU pragma: keep
#include "llm/npu/glm4.h"                     // IWYU pragma: keep
#include "llm/npu/glm4_moe.h"                 // IWYU pragma: keep
#include "llm/npu/glm4_moe_lite.h"            // IWYU pragma: keep
#include "llm/npu/glm4_moe_mtp.h"             // IWYU pragma: keep
#include "llm/npu/glm5_moe.h"                 // IWYU pragma: keep
#include "llm/npu/glm5_moe_mtp.h"             // IWYU pragma: keep
#include "llm/npu/kimi_k2.h"                  // IWYU pragma: keep
#include "llm/npu/llama.h"                    // IWYU pragma: keep
#include "llm/npu/llama3.h"                   // IWYU pragma: keep
#include "llm/npu/custom_qwen3_npu.h"          // IWYU pragma: keep
#include "llm/npu/qwen2.h"                    // IWYU pragma: keep
#include "llm/npu/qwen3.h"                    // IWYU pragma: keep
#include "llm/npu/qwen3_eagle3.h"             // IWYU pragma: keep
#include "llm/npu/qwen3_moe.h"                // IWYU pragma: keep
#include "rec/onerec.h"                       // IWYU pragma: keep
#include "vlm/npu/glm4v.h"                    // IWYU pragma: keep
#include "vlm/npu/glm4v_moe.h"                // IWYU pragma: keep
#include "vlm/npu/minicpmv.h"                 // IWYU pragma: keep
#include "vlm/npu/custom_qwen3_npu_vlm.h"      // IWYU pragma: keep
#include "vlm/npu/qwen2_5_vl.h"               // IWYU pragma: keep
#include "vlm/npu/qwen2_5_vl_mm_embedding.h"  // IWYU pragma: keep
#include "vlm/npu/qwen2_vl.h"                 // IWYU pragma: keep
#include "vlm/npu/qwen2_vl_embedding.h"       // IWYU pragma: keep
#include "vlm/npu/qwen3_vl.h"                 // IWYU pragma: keep
#include "vlm/npu/qwen3_vl_mm_embedding.h"    // IWYU pragma: keep
#include "vlm/npu/qwen3_vl_moe.h"             // IWYU pragma: keep
#elif defined(USE_MLU)
#include "llm/deepseek_mtp.h"        // IWYU pragma: keep
#include "llm/deepseek_v2.h"         // IWYU pragma: keep
#include "llm/deepseek_v3.h"         // IWYU pragma: keep
#include "llm/deepseek_v32.h"        // IWYU pragma: keep
#include "llm/glm5.h"                // IWYU pragma: keep
#include "llm/glm5_mtp.h"            // IWYU pragma: keep
#include "llm/custom_moe_flash.h"    // IWYU pragma: keep
#include "llm/mtp_model_base.h"      // IWYU pragma: keep
#include "llm/qwen2.h"               // IWYU pragma: keep
#include "llm/qwen3.h"               // IWYU pragma: keep
#include "llm/qwen3_moe.h"           // IWYU pragma: keep
#include "vlm/qwen2_5_vl.h"          // IWYU pragma: keep
#include "vlm/qwen2_vl.h"            // IWYU pragma: keep
#include "vlm/qwen2_vl_embedding.h"  // IWYU pragma: keep
#include "vlm/qwen3_vl.h"            // IWYU pragma: keep
#include "vlm/qwen3_vl_moe.h"        // IWYU pragma: keep
#elif defined(USE_ILU)
#include "llm/qwen2.h"      // IWYU pragma: keep
#include "llm/qwen3.h"      // IWYU pragma: keep
#include "llm/qwen3_moe.h"  // IWYU pragma: keep
#elif defined(USE_CUDA)
#include "dit/pipeline_longcat_image.h"       // IWYU pragma: keep
#include "dit/pipeline_longcat_image_edit.h"  // IWYU pragma: keep
#include "llm/qwen2.h"                        // IWYU pragma: keep
#include "llm/qwen3.h"                        // IWYU pragma: keep
#include "llm/qwen3_moe.h"                    // IWYU pragma: keep
#include "vlm/qwen2_5_vl.h"                   // IWYU pragma: keep
#include "vlm/qwen2_vl.h"                     // IWYU pragma: keep
#include "vlm/qwen2_vl_embedding.h"           // IWYU pragma: keep
#include "vlm/qwen3_vl.h"                     // IWYU pragma: keep
#include "vlm/qwen3_vl_moe.h"                 // IWYU pragma: keep
#elif defined(USE_MUSA)
#include "llm/musa/qwen3.h"  // IWYU pragma: keep
#else
#error "Unsupported device type, only support NPU, CUDA, MLU, ILU and MUSA now."
#endif
