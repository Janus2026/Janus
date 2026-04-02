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

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexGroupMatmulTilingData)
TILING_DATA_FIELD_DEF(uint32_t, M);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, K);
TILING_DATA_FIELD_DEF(uint32_t, baseM);
TILING_DATA_FIELD_DEF(uint32_t, baseN);
TILING_DATA_FIELD_DEF(uint32_t, baseK);
TILING_DATA_FIELD_DEF(uint32_t, tailM);
TILING_DATA_FIELD_DEF(uint32_t, tailK);
TILING_DATA_FIELD_DEF(uint32_t, tailN);
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, actExperts);
TILING_DATA_FIELD_DEF(uint32_t, isBf16);
// TILING_DATA_FIELD_DEF_ARR(int64_t,256,groupOffset);
// TILING_DATA_FIELD_DEF_ARR(int32_t,416,sortedList);


END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexGroupMatmul, IndexGroupMatmulTilingData)
} // namespace optiling
