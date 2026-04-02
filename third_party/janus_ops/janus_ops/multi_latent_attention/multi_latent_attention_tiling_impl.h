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

#ifndef MLA_TILING_H
#define MLA_TILING_H

#include "multi_latent_attention_tiling_dependency.h"
#include "mla.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/tiling_api.h"
 
namespace AtbOps {
ge::graphStatus MLATiling(gert::TilingContext *context);
ge::graphStatus GetMLATilingParam(OpParam::MLA param, const MLAInfo &mmInfo,
    uint32_t &blockDim, uint32_t *tilingParam, uint64_t tilingParamSize);
}

#endif // MLA_TILING_H
