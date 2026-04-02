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
BEGIN_TILING_DATA_DEF(SelectUnsharedKVTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, total_beam);
  TILING_DATA_FIELD_DEF(uint32_t, head_num);
  TILING_DATA_FIELD_DEF(uint32_t, head_dim);
  TILING_DATA_FIELD_DEF(uint32_t, copy_head_num_per_loop);
  TILING_DATA_FIELD_DEF(uint32_t, copy_repeat_times);
  TILING_DATA_FIELD_DEF(uint32_t, copy_head_num_tail);
  TILING_DATA_FIELD_DEF(uint32_t, max_decode_step);
  TILING_DATA_FIELD_DEF(uint32_t, used_core_num);
  TILING_DATA_FIELD_DEF(uint64_t, block_beam_stride);
  TILING_DATA_FIELD_DEF(uint64_t, block_batch_stride);
  TILING_DATA_FIELD_DEF(uint64_t, block_head_stride);
  TILING_DATA_FIELD_DEF(uint32_t, decode_step);
  TILING_DATA_FIELD_DEF(uint32_t, beam_size);
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, layer_num);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SelectUnsharedKV, SelectUnsharedKVTilingData)
}
