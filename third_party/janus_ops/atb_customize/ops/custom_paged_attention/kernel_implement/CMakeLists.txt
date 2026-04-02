# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(custom_paged_attention_srcs
    ${CMAKE_CURRENT_LIST_DIR}/custom_paged_attention_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/custom_paged_attention_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/custom_paged_attention_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/custom_paged_attention_tiling_dependency.cpp
)

add_operation(CustomPagedAttentionOperation "${custom_paged_attention_srcs}")

add_kernel(custom_paged_attention_mask ascend910b mix
    op_kernel/custom_paged_attention_mask_mix.cce
    CustomPagedAttentionMaskNdKernel)
