/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 /*!
 * \file moe_fused_add_topk.cpp
 * \brief
 */

#include "moe_fused_add_topk.h"
#include "kernel_operator.h"
using namespace AscendC;

#define INVOKE_MOE_FUSED_ADD_TOPK_IMPL(templateClass, ...)                                                  \
    do {                                                                                                        \
        templateClass<__VA_ARGS__> op;                                                                          \
        op.InitTilingData(&tilingData, x, addNum, mappingNum, mappingTable, y, indices, userWsp);               \
        op.InitBuffer(&pipe);                                                                                   \
        op.Process();                                                                                           \
    } while (0)

extern "C" __global__ __aicore__ void moe_fused_add_topk(GM_ADDR x, GM_ADDR addNum,
                                                         GM_ADDR mappingNum, GM_ADDR mappingTable, GM_ADDR y,
                                                         GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    PRELOAD(8);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWsp = GetUserWorkspace(workspace);
    if (userWsp == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

#if (ORIG_DTYPE_X == DT_FLOAT)
    if (TILING_KEY_IS(0)) {
        INVOKE_MOE_FUSED_ADD_TOPK_IMPL(MoeFusedAddTopk, float, float, 0);
        return;
    } else if (TILING_KEY_IS(1)) {
        INVOKE_MOE_FUSED_ADD_TOPK_IMPL(MoeFusedAddTopk, float, float, 1);
        return;
    }
#endif
#if (ORIG_DTYPE_X == DT_FLOAT16)
    if (TILING_KEY_IS(0)) {
        INVOKE_MOE_FUSED_ADD_TOPK_IMPL(MoeFusedAddTopk, half, float, 0);
        return;
    } else if (TILING_KEY_IS(1)) {
        INVOKE_MOE_FUSED_ADD_TOPK_IMPL(MoeFusedAddTopk, half, float, 1);
        return;
    }
#endif
#if (ORIG_DTYPE_X == DT_BF16)
    if (TILING_KEY_IS(0)) {
        INVOKE_MOE_FUSED_ADD_TOPK_IMPL(MoeFusedAddTopk, bfloat16_t, float, 0);
        return;
    } else if (TILING_KEY_IS(1)) {
        INVOKE_MOE_FUSED_ADD_TOPK_IMPL(MoeFusedAddTopk, bfloat16_t, float, 1);
        return;
    }
#endif
}
