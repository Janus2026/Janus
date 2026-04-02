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

#include "convert_kv_cache_format_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  ConvertKvCacheFormatTilingData tiling;

  // [num_batches] (int64)
  auto kv_cache_offset_shape = context->GetInputShape(2)->GetOriginShape();
  auto num_batches = kv_cache_offset_shape.GetDim(0);

  auto is_prefill = *(context->GetAttrs()->GetBool(0));
  auto num_kv_heads = *(context->GetAttrs()->GetInt(1));
  auto head_size_k = *(context->GetAttrs()->GetInt(2));
  auto head_size_v = *(context->GetAttrs()->GetInt(3));

  tiling.set_is_prefill(static_cast<uint32_t>(is_prefill));
  tiling.set_num_batches(static_cast<uint32_t>(num_batches));
  tiling.set_num_kv_heads(static_cast<uint32_t>(num_kv_heads));
  tiling.set_head_size_k(static_cast<uint32_t>(head_size_k));
  tiling.set_head_size_v(static_cast<uint32_t>(head_size_v));
  tiling.set_block_size(128);

  auto platform_info = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t aiv_num = platform_info.GetCoreNumAiv();
  context->SetBlockDim(aiv_num);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  size_t systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = systemWorkspaceSize;

  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ConvertKvCacheFormat : public OpDef {
 public:
  explicit ConvertKvCacheFormat(const char* name) : OpDef(name) {
    this->Input("k_cache_ptr")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("v_cache_ptr")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("kv_cache_offset")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("kv_seq_len")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->Attr("is_prefill").Bool();
    this->Attr("num_kv_heads").Int();
    this->Attr("head_size_k").Int();
    this->Attr("head_size_v").Int();

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(ConvertKvCacheFormat);
}  // namespace ops
