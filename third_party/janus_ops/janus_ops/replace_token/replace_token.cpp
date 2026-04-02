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

#include "./replace_token.h"

#include "kernel_operator.h"

extern "C" __global__ __aicore__ void replace_token(GM_ADDR forkedTokenIds,
                                                    GM_ADDR lastStepOutPutTokenIds,
                                                    GM_ADDR out,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    ReplaceToken op;
    op.Init(forkedTokenIds,
            lastStepOutPutTokenIds,
            out,
            tiling_data.sequenceLength,
            tiling_data.blength,
            tiling_data.max_tokens);
    op.Process();
}
