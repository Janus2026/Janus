/* Copyright 2025 The Janus Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at


Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "index_group_matmul.h"
#include "index_group_matmul_mlt512.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

//using namespace matmul;

extern "C" __global__ __aicore__ void
index_group_matmul(GM_ADDR a, GM_ADDR b, GM_ADDR scale, GM_ADDR perTokenScale,
                   GM_ADDR groupList, GM_ADDR c, GM_ADDR workSpace,
                   GM_ADDR tiling) {

  // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY); //910B should be enabled to
  // use only Cube cores KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  GET_TILING_DATA(tiling_data, tiling);

  if (TILING_KEY_IS(0)) {
    // if  m is 13 and 1,n is 7168, k is 128
    KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    KernelMatmulInt128Mlt512 op;
    op.Init(a, b, scale, perTokenScale, groupList, c, workSpace, tiling_data.M,
            tiling_data.N, tiling_data.K, tiling_data.baseM, tiling_data.baseN,
            tiling_data.baseK, tiling_data.tailM, tiling_data.tailN,
            tiling_data.tailK, tiling_data.groupNum, tiling_data.actExperts);
    op.Process();
  } else if (TILING_KEY_IS(1)) {
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_1);
    KernelMatmulInt128 op;
    op.Init(a, b, scale, perTokenScale, groupList, c, workSpace, tiling_data.M,
            tiling_data.N, tiling_data.K, tiling_data.baseM, tiling_data.baseN,
            tiling_data.baseK, tiling_data.tailM, tiling_data.tailN,
            tiling_data.tailK, tiling_data.groupNum, tiling_data.actExperts);
  }
}
