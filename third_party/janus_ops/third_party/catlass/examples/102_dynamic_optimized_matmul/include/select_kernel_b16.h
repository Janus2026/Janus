/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECT_KERNEL_HALF_H
#define SELECT_KERNEL_HALF_H

#include "platform_info.h"

bool CommonMatmulB16Handler(TilingParams &params, PlatformInfo& platformInfo)
{
    uint8_t kernelSerial = 0;
    uint32_t taskBlocks = CeilDiv(params.m, params.m1) * CeilDiv(params.n, params.n1);
    params.blockDim = taskBlocks > platformInfo.coreNum ? platformInfo.coreNum : taskBlocks;

    // kernelSerial, layoutTagA, layoutTagB, layoutTagC, paddingTagA, paddingTagB, paddingTagC, dtype(defalut 0).
    params.tilingKey.SetTilingKey(kernelSerial, params.layoutTagA, params.layoutTagB, 0, 0, 0, 0);
    return true;
}

void SelectKernelB16(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    // Temporarily store the original layoutTagA and layoutTagB
    uint8_t layoutTagATmp = tilingParams.layoutTagA;
    uint8_t layoutTagBTmp = tilingParams.layoutTagB;
    // When m=1 or n=1, the row-major and column-major matrix layouts are indentical, the matrix can be stored
    // in either format. In such cases, the layout with higher memory transfer bandwidth should be selected.
    if (tilingParams.m == 1 && static_cast<LayoutTag>(tilingParams.layoutTagA) == LayoutTag::TagColumnMajor) {
        tilingParams.layoutTagA = static_cast<uint8_t>(LayoutTag::TagRowMajor);
    }
    if (tilingParams.n == 1 && static_cast<LayoutTag>(tilingParams.layoutTagB) == LayoutTag::TagRowMajor) {
        tilingParams.layoutTagB = static_cast<uint8_t>(LayoutTag::TagColumnMajor);
    }

    using HandlerPtr = bool (*)(TilingParams& tilingParams, PlatformInfo& platformInfo);
    HandlerPtr handlers[] = {
        CommonMatmulB16Handler
    };

    for (auto handler : handlers) {
        if (handler(tilingParams, platformInfo)) {
            break;
        }
    }

    // Restore to the original layout
    tilingParams.layoutTagA = layoutTagATmp;
    tilingParams.layoutTagB = layoutTagBTmp;
}

#endif  // SELECT_KERNEL_HALF_H