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

#include "register/op_def_registry.h"
#include "replace_token_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    ReplaceTokenTilingData tiling;
    auto aShape = context->GetInputShape(0)->GetOriginShape();
    auto bShape = context->GetInputShape(1)->GetOriginShape();
    size_t aDim = aShape.GetDimNum();
    size_t bDim = bShape.GetDimNum();
    uint32_t max_tokens = 10000;
    int32_t sequenceLength = aShape.GetDim(0);
    int32_t blength = bShape.GetDim(0);
    tiling.set_sequenceLength(sequenceLength);
    tiling.set_max_tokens(max_tokens);
    tiling.set_blength(blength);
    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class ReplaceToken : public OpDef {
  public:
    explicit ReplaceToken(const char* name) : OpDef(name) {
        this->Input("forkedTokenIds")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("lastStepOutPutTokenIds")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT64 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("Out")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(ReplaceToken);
} // namespace ops
