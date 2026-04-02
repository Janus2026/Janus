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

#include "kernel_operator.h"
#include "x_attention_catlass_helper.h"
#include "lib/matmul_intf.h"

#define CALL_XATTN_KERNEL(INPUT_TYPE, SHARED_PAGED_FLAG, UNSHARED_PAGED_FLAG) \
    do { \
        if (coreIdx < tiling_data.sharedCoreNum) { \
            CallSharedInferKernelShort<INPUT_TYPE, SHARED_PAGED_FLAG>(params, &tiling_data); \
        } else { \
            CallUnsharedInferKernel<INPUT_TYPE, UNSHARED_PAGED_FLAG>(params, &tiling_data); \
        } \
        AscendC::SyncAll<false>(); \
        CallCombineScale<INPUT_TYPE>(params, &tiling_data); \
    } while (0)

using namespace AscendC;

extern "C" __global__ __aicore__ void x_attention(GM_ADDR query, GM_ADDR shared_key_block, GM_ADDR shared_value_block, 
                        GM_ADDR unshared_key_block, GM_ADDR unshared_value_block, GM_ADDR unshared_block_table,
                        GM_ADDR shared_kv_lens, GM_ADDR decode_step, GM_ADDR shared_block_table, GM_ADDR attn_out, GM_ADDR workspace, GM_ADDR tiling) {
    // workspace use; [s,p,oTemp,oUpdate,shared_workspace,unshared_workspace]
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);

    GM_ADDR s = workspace;
    GM_ADDR p = workspace + tiling_data.mm1OutSize;
    GM_ADDR oTemp = p + tiling_data.smOnlineOutSize;
    GM_ADDR oUpdate = oTemp + tiling_data.mm2OutSize;
    GM_ADDR shared_workspace = oUpdate + tiling_data.updateSize;
    GM_ADDR unshared_workspace = shared_workspace + tiling_data.sharedWorkspaceSize;
    int64_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

    XAttnKernelParams params{query, shared_key_block, shared_value_block, unshared_key_block, unshared_value_block, 
     shared_block_table, unshared_block_table, shared_kv_lens, decode_step, s, p, oTemp, oUpdate, shared_workspace, 
         unshared_workspace, attn_out, tiling};
    if (TILING_KEY_IS(4)) {          // 0b0100
        CALL_XATTN_KERNEL(half, false, true);
    } else if (TILING_KEY_IS(6)) {   // 0b0110
        CALL_XATTN_KERNEL(bfloat16_t, false, true);
    } else if (TILING_KEY_IS(8)) {   // 0b1000
        CALL_XATTN_KERNEL(half, true, false);
    } else if (TILING_KEY_IS(10)) {  // 0b1010
        CALL_XATTN_KERNEL(bfloat16_t, true, false);
    }
}
