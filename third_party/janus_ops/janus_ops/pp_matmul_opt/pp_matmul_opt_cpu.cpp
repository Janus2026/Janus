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

#include "pp_matmul_opt_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {

  PpMatmulOptTilingData tiling;

  auto a_shape = context->GetInputShape(0)->GetOriginShape();
  auto b_shape = context->GetInputShape(1)->GetOriginShape();
  auto c_shape = context->GetOutputShape(0)->GetOriginShape();
  size_t a_dim = a_shape.GetDimNum();
  size_t b_dim = b_shape.GetDimNum();
  size_t c_dim = c_shape.GetDimNum();
  uint32_t M = a_shape.GetDim(a_dim - 2);
  uint32_t N = b_shape.GetDim(b_dim - 2);
  uint32_t K = a_shape.GetDim(a_dim - 1);

  constexpr auto m0 = 16;
  constexpr auto k0 = 256;
  constexpr auto n0 = 256;
  // 64KB / pingpong_buffer_num / sizeof(bf16) / n0
  constexpr auto kPart = 64 * 1024 / 2 / 2 / n0;
  constexpr auto mmadOffsetMax = (k0 + kPart - 1) / kPart * kPart * 16;

  tiling.set_m(M);
  tiling.set_k(K);
  tiling.set_n(N);
  tiling.set_rows(K / 256);
  tiling.set_mmadOffsetMax(mmadOffsetMax);

  context->SetBlockDim(24);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  size_t systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = systemWorkspaceSize + M * N * sizeof(float) + 16 * 16 * sizeof(float);

  return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Shape *a_shape = context->GetInputShape(0);
  const gert::Shape *b_shape = context->GetInputShape(1);
  gert::Shape *c_shape = context->GetOutputShape(0);
  *c_shape = *a_shape;
  size_t b_dim = b_shape->GetDimNum();
  size_t c_dim = c_shape->GetDimNum();
  uint32_t N = b_shape->GetDim(b_dim - 2);
  c_shape->SetDim(c_dim - 1, N);
  return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class PpMatmulOpt : public OpDef {
public:
  explicit PpMatmulOpt(const char *name) : OpDef(name) {
    this->Input("a")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("b")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("c")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(PpMatmulOpt);
} // namespace ops
