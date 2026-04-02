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
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvertKvCacheFormatTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, is_prefill);
  TILING_DATA_FIELD_DEF(uint32_t, num_batches);
  TILING_DATA_FIELD_DEF(uint32_t, num_kv_heads);
  TILING_DATA_FIELD_DEF(uint32_t, head_size_k);
  TILING_DATA_FIELD_DEF(uint32_t, head_size_v);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvertKvCacheFormat, ConvertKvCacheFormatTilingData)
}