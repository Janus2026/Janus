/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "x_flash_attention_infer.h"
#include "x_flash_attention_infer_fd.h"

extern "C" __global__ __aicore__ void x_flash_attention_infer(GM_ADDR query, GM_ADDR key_cache, GM_ADDR value_cache,
                        GM_ADDR mask, GM_ADDR block_table, GM_ADDR actual_q_lens,
                        GM_ADDR actual_kv_lens, GM_ADDR extra_tiling, GM_ADDR attn_out, GM_ADDR workspace, GM_ADDR tiling) {
    // workspace use; [s,p,oTemp,oUpdate,shared_workspace,unshared_workspace]
    SetAtomicNone();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);

    GM_ADDR s = workspace;
    GM_ADDR p = workspace + tiling_data.mm1OutSize;
    GM_ADDR oTemp = p + tiling_data.smOnlineOutSize;
    GM_ADDR oUpdate = oTemp + tiling_data.mm2OutSize;
    GM_ADDR gmlse = oUpdate + tiling_data.updateSize;
    GM_ADDR glo = gmlse + tiling_data.splitLseTotalSize;
    __gm__ SplitKvExtraInfo *extraInfo = reinterpret_cast<__gm__ SplitKvExtraInfo *>(extra_tiling);
    auto coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    if (TILING_KEY_IS(QFP16_KVFP16_TND_CAUSALMASK_FD_TILING)) {  // FD fp16
        if (extraInfo->coreInfo[coreIdx].startBIdx != UINT32_MAX) {
            FDInfer<half, half, layout::ColumnMajor, layout::RowMajor ,true, FaiKenel::MaskType::MASK_CAUSAL>(query, key_cache, value_cache, mask, block_table,
            attn_out, actual_q_lens, actual_kv_lens, s, p, oTemp, oUpdate, gmlse, glo, tiling, extraInfo);
        } else {
            AscendC::SyncAll();
        }
    } else if (TILING_KEY_IS(QBF16_KVBF16_TND_CAUSALMASK_FD_TILING)) {  // FD bf16
        if (extraInfo->coreInfo[coreIdx].startBIdx != UINT32_MAX) {
            FDInfer<bfloat16_t, bfloat16_t, layout::ColumnMajor, layout::RowMajor, true, FaiKenel::MaskType::MASK_CAUSAL>(query, key_cache, value_cache, mask, block_table,
            attn_out, actual_q_lens, actual_kv_lens, s, p, oTemp, oUpdate, gmlse, glo, tiling, extraInfo);
        } else {
            AscendC::SyncAll();
        }
    } else if (TILING_KEY_IS(QFP16_KVFP16_KVNZ_CAUSALMASK_FD_TILING)) {
        if (extraInfo->coreInfo[coreIdx].startBIdx != UINT32_MAX) {
            FDInfer<half, half, layout::nZ, layout::zN, true, FaiKenel::MaskType::MASK_CAUSAL>(query, key_cache, value_cache, mask, block_table,
            attn_out, actual_q_lens, actual_kv_lens, s, p, oTemp, oUpdate, gmlse, glo, tiling, extraInfo);
        } else {
            AscendC::SyncAll();
        }
    } else if (TILING_KEY_IS(QBF16_KVBF16_KVNZ_CAUSALMASK_FD_TILING)) {
        if (extraInfo->coreInfo[coreIdx].startBIdx != UINT32_MAX) {
            FDInfer<bfloat16_t, bfloat16_t, layout::nZ, layout::zN, true, FaiKenel::MaskType::MASK_CAUSAL>(query, key_cache, value_cache, mask, block_table,
            attn_out, actual_q_lens, actual_kv_lens, s, p, oTemp, oUpdate, gmlse, glo, tiling, extraInfo);
        } else {
            AscendC::SyncAll();
        }
    }
}
