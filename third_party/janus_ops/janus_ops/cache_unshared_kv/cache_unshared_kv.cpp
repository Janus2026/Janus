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
#include "cache_unshared_kv.h"

extern "C" __global__ __aicore__ void cache_unshared_kv(GM_ADDR x_key_block, GM_ADDR x_value_block, 
        GM_ADDR curr_key, GM_ADDR curr_value, GM_ADDR block_table, GM_ADDR decode_step, 
        GM_ADDR select_key_block, GM_ADDR select_value_block, 
        GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe pipe;
    kernels::CacheUnsharedKvKernel<DTYPE_X_KEY_BLOCK> op(&pipe);
    op.Init(x_key_block, x_value_block, curr_key, curr_value, block_table, decode_step, &tiling_data, select_key_block, select_value_block);
    op.process();
}
