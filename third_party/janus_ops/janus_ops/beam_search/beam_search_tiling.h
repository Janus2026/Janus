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
    
BEGIN_TILING_DATA_DEF(BeamSearchTilingData)
TILING_DATA_FIELD_DEF(int32_t, num_sequences);
TILING_DATA_FIELD_DEF(int32_t, sequence_length);
TILING_DATA_FIELD_DEF(int32_t, beam_width);
TILING_DATA_FIELD_DEF(int32_t, top_k);
TILING_DATA_FIELD_DEF(int32_t, request_num);
TILING_DATA_FIELD_DEF(int32_t, core_num);
TILING_DATA_FIELD_DEF(int32_t, min_size);
TILING_DATA_FIELD_DEF(int32_t, step_size);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topKTilingData1);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BeamSearch, BeamSearchTilingData)

}  // namespace optiling