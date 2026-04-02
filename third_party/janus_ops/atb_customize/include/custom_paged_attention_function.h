/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <cstdint>
#include <string>
#include <limits>
#include <acl/acl.h>
#include "atb/operation.h"

namespace atb {
namespace customize {

struct TilingBufferInfo {
    uint8_t* tilingBuffer;
    uint64_t tilingBufferSize;
};

TilingBufferInfo GetHostTilingBufferFromCustomPagedAttentionOperation(const Operation *operation);

Status CreatePlanContext(Context **context);

} // namespace customize
} // namespace atb
