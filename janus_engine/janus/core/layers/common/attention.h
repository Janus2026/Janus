#pragma once

#if defined(USE_MLU)
#include "layers/mlu/attention.h"
#elif defined(USE_NPU)
#include "layers/npu_torch/attention.h"
#elif defined(USE_CUDA)
#include "layers/cuda/attention.h"
#elif defined(USE_ILU)
#include "layers/ilu/attention.h"
#elif defined(USE_MUSA)
#include "layers/musa/attention.h"
#endif
