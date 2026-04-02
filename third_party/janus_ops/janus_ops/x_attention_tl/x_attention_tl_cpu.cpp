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

#include "x_attention_tl_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define OP_LOGD(nodeName, fmt, ...) \
  printf(fmt, ##__VA_ARGS__);       \
  printf("\n")
#define OP_LOGE(nodeName, fmt, ...) \
  printf(fmt, ##__VA_ARGS__);       \
  printf("\n")

namespace optiling {
class TilingXAttentionTlFunc {
 public:
  explicit TilingXAttentionTlFunc(gert::TilingContext* tiling_context)
      : tiling_context_(tiling_context) {}

  ge::graphStatus Init();
  ge::graphStatus RunKernelTiling();

 private:
  XAttentionTlTilingData tiling_data_;
  gert::TilingContext* tiling_context_ = nullptr;

  void SetTilingKey();
  void FillTilingData();
  // TODO: uint32_t to int32_t
  uint32_t batch_size_ = 0;
  uint32_t num_heads_ = 0;
  uint32_t head_size_ = 0;
  uint32_t q_length_ = 0;
  uint32_t unshared_k_length_ = 0;
  uint32_t shared_k_length_ = 0;
  uint32_t beam_size_ = 0;
  uint32_t core_num_ = 0;
  size_t sync_workspace_size_ = 0;
};

ge::graphStatus TilingXAttentionTlFunc::Init() {
  auto platform_info =
      platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
  uint32_t aic_num = platform_info.GetCoreNumAic();
  // uint32_t aic_num = 8;
  uint32_t aiv_num = platform_info.GetCoreNumAiv();
  // uint32_t aiv_num = 1;

  // check input shape
  auto query_shape = tiling_context_->GetInputShape(0)->GetOriginShape();
  auto shared_key_shape = tiling_context_->GetInputShape(1)->GetOriginShape();
  auto unshared_key_shape = tiling_context_->GetInputShape(4)->GetOriginShape();
  auto query_unshared_shape = tiling_context_->GetInputShape(3)->GetOriginShape();

  batch_size_ = query_shape.GetDim(query_shape.GetDimNum() - 4);
  num_heads_ = query_shape.GetDim(query_shape.GetDimNum() - 3);
  beam_size_ = query_shape.GetDim(query_shape.GetDimNum() - 2);
  head_size_ = query_shape.GetDim(query_shape.GetDimNum() - 1);
  shared_k_length_ = shared_key_shape.GetDim(shared_key_shape.GetDimNum() - 2);
  q_length_ = query_unshared_shape.GetDim(query_unshared_shape.GetDimNum() - 2);
  unshared_k_length_ = unshared_key_shape.GetDim(unshared_key_shape.GetDimNum() - 2);
  core_num_ = aic_num;
  sync_workspace_size_ =
      static_cast<size_t>(platform_info.GetLibApiWorkSpaceSize());
  return ge::GRAPH_SUCCESS;
}

void TilingXAttentionTlFunc::SetTilingKey() { tiling_context_->SetTilingKey(0); }

void TilingXAttentionTlFunc::FillTilingData() {
  tiling_data_.set_batch_size(batch_size_);
  tiling_data_.set_num_heads(num_heads_);
  tiling_data_.set_head_size(head_size_);
  tiling_data_.set_q_length(q_length_);
  tiling_data_.set_unshared_k_length(unshared_k_length_);
  tiling_data_.set_shared_k_length(shared_k_length_);
  tiling_data_.set_beam_size(beam_size_);
  tiling_data_.set_core_num(core_num_);
}

ge::graphStatus TilingXAttentionTlFunc::RunKernelTiling() {
  SetTilingKey();
  FillTilingData();
  size_t userWorkspaceSize = batch_size_ * num_heads_ *  head_size_  * beam_size_ * sizeof(float) * 6;
  userWorkspaceSize += batch_size_ * num_heads_ *  head_size_ * beam_size_ * shared_k_length_ * sizeof(float) * 6;
  size_t* currentWorkspace = tiling_context_->GetWorkspaceSizes(1);
  currentWorkspace[0] = userWorkspaceSize + sync_workspace_size_;
  tiling_data_.SaveToBuffer(tiling_context_->GetRawTilingData()->GetData(),
                            tiling_context_->GetRawTilingData()->GetCapacity());
  tiling_context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
  tiling_context_->SetBlockDim(core_num_);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForXAttentionTlFunc(gert::TilingContext* context) {
  TilingXAttentionTlFunc tilingObject(context);
  auto ret = tilingObject.Init();
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(context->GetNodeName(), "tiling Init failed.");
    return ge::GRAPH_FAILED;
  }
  ret = tilingObject.RunKernelTiling();
  return ret;
}

}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x1_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class XAttentionTl : public OpDef {
 public:
  explicit XAttentionTl(const char* name) : OpDef(name) {
    this->Input("q_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("k_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("v_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("q_unshared_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("unshared_k_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("unshared_v_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("output_shared_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("output_unshared_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("shared_exp_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("unshared_exp_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("shared_max_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("unshared_max_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Output("output")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->SetInferShape(ge::InferShape);
    this->AICore().SetTiling(optiling::TilingForXAttentionTlFunc);
    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(XAttentionTl);
}  // namespace ops
