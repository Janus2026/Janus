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

#include "index_group_matmul_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define OP_LOGD(nodeName, fmt, ...) printf(fmt, ##__VA_ARGS__); printf("\n")
#define OP_LOGE(nodeName, fmt, ...) printf(fmt, ##__VA_ARGS__); printf("\n")

namespace optiling {
class TilingGroupMatmulFunc {
 public:
  explicit TilingGroupMatmulFunc(gert::TilingContext* tiling_context)
      : tiling_context_(tiling_context) {}
  
  ge::graphStatus Init();
  ge::graphStatus RunKernelTiling();

 private:
  IndexGroupMatmulTilingData tiling_data_;
  gert::TilingContext* tiling_context_ = nullptr;
  
  void SetTilingKey();
  void FillTilingData();
  
  uint32_t m_ = 0;
  uint32_t n_ = 0;
  uint32_t k_ = 0;
  uint32_t base_m_ = 0;
  uint32_t base_n_ = 0;
  uint32_t base_k_ = 0;
  uint32_t tile_m_ = 0;
  uint32_t tile_n_ = 0;
  uint32_t tile_k_ = 0;
  uint32_t group_num_ = 0;
  uint32_t is_bf16_ = 0;
  uint32_t core_num_ = 0;
  uint32_t act_experts_ = 0;
  size_t sync_workspace_size_ = 0;
};

ge::graphStatus TilingGroupMatmulFunc::Init() {
  auto platform_info = platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
  uint32_t aic_num = platform_info.GetCoreNumAic();
  uint32_t aiv_num = platform_info.GetCoreNumAiv();
  uint64_t platform_ub_size = 0;
  
  //check input shape 
  auto a_shape = tiling_context_->GetInputShape(0)->GetOriginShape();
  auto b_shape = tiling_context_->GetInputShape(1)->GetOriginShape();
  auto scale_shape = tiling_context_->GetInputShape(2)->GetOriginShape();
  auto per_token_scale_shape = tiling_context_->GetInputShape(3)->GetOriginShape();
  auto group_list_shape = tiling_context_->GetInputShape(4)->GetOriginShape();
  auto c_shape = tiling_context_->GetOutputShape(0)->GetOriginShape();

  m_ = a_shape.GetDim(a_shape.GetDimNum() - 2);
  n_ = b_shape.GetDim(b_shape.GetDimNum() - 1);
  k_ = a_shape.GetDim(a_shape.GetDimNum() - 1);
  group_num_ = b_shape.GetDim(b_shape.GetDimNum() - 3);
  core_num_ = aic_num;
  act_experts_ = c_shape.GetDim(c_shape.GetDimNum() - 2)/m_;
  base_m_ = 16;
  base_n_ = 256;
  base_k_ = 128;
  sync_workspace_size_ = static_cast<size_t>(platform_info.GetLibApiWorkSpaceSize());

  if(a_shape.GetDimNum() != 2 || b_shape.GetDimNum() != 3 || scale_shape.GetDimNum() != 2 || per_token_scale_shape.GetDimNum() != 1 || group_list_shape.GetDimNum() != 1 || c_shape.GetDimNum() != 2){
    OP_LOGE(tiling_context_->GetNodeName(), "the dimNum of input and output should be 2, but got %zu, %zu, %zu, %zu, %zu, %zu.", static_cast<size_t>(a_shape.GetDimNum()), static_cast<size_t>(b_shape.GetDimNum()), static_cast<size_t>(scale_shape.GetDimNum()), static_cast<size_t>(per_token_scale_shape.GetDimNum()), static_cast<size_t>(group_list_shape.GetDimNum()), static_cast<size_t>(c_shape.GetDimNum()));
    return ge::GRAPH_FAILED;
  }
  if(a_shape.GetDim(1) != b_shape.GetDim(1)){
    OP_LOGE(tiling_context_->GetNodeName(), "the dim of input a and b should be equal, but got %zu, %zu.", static_cast<size_t>(a_shape.GetDim(1)), static_cast<size_t>(b_shape.GetDim(1)));
    return ge::GRAPH_FAILED;
  }
  if(scale_shape.GetDim(0) != group_num_ || scale_shape.GetDim(1) != n_){
    OP_LOGE(tiling_context_->GetNodeName(), "the dim of input scale should be equal to input a and b, but got %zu, %zu.", static_cast<size_t>(scale_shape.GetDim(0)), static_cast<size_t>(scale_shape.GetDim(1)));
    return ge::GRAPH_FAILED;
  }
  if(per_token_scale_shape.GetDim(0) != m_){
    OP_LOGE(tiling_context_->GetNodeName(), "the dim of input per_token_scale should be equal to input a, but got %zu.", static_cast<size_t>(per_token_scale_shape.GetDim(0)));
    return ge::GRAPH_FAILED;
  }
  if(group_list_shape.GetDim(0) != group_num_){
    OP_LOGE(tiling_context_->GetNodeName(), "the dim of input group_list should be equal to input b, but got %zu.", static_cast<size_t>(group_list_shape.GetDim(0)));
    return ge::GRAPH_FAILED;
  }
  if(c_shape.GetDim(c_shape.GetDimNum() - 2) != m_){
    OP_LOGE(tiling_context_->GetNodeName(), "the dim of input c should be equal to input a, but got %zu.", static_cast<size_t>(c_shape.GetDim(c_shape.GetDimNum() - 2)));
    return ge::GRAPH_FAILED;
  }
  if(c_shape.GetDim(c_shape.GetDimNum() - 1) != n_){
    OP_LOGE(tiling_context_->GetNodeName(), "the dim of input c should be equal to input b, but got %zu.", static_cast<size_t>(c_shape.GetDim(c_shape.GetDimNum() - 1)));
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

void TilingGroupMatmulFunc::SetTilingKey() {
  tiling_context_->SetTilingKey(0);
}

void TilingGroupMatmulFunc::FillTilingData() {
  tiling_data_.set_M(m_);
  tiling_data_.set_N(n_);
  tiling_data_.set_K(k_);
  tiling_data_.set_baseM(base_m_);
  tiling_data_.set_baseN(base_n_);
  tiling_data_.set_baseK(base_k_);
  tiling_data_.set_tailM(tile_m_);
  tiling_data_.set_tailN(tile_n_);
  tiling_data_.set_tailK(tile_k_);
  tiling_data_.set_groupNum(group_num_);
  tiling_data_.set_actExperts(act_experts_);
}

ge::graphStatus TilingGroupMatmulFunc::RunKernelTiling() {
  SetTilingKey();
  FillTilingData();
  size_t userWorkspaceSize = 8*m_ *  n_ * sizeof(int32_t);
  size_t *currentWorkspace = tiling_context_->GetWorkspaceSizes(1);
  currentWorkspace[0] = userWorkspaceSize + sync_workspace_size_;
  tiling_data_.SaveToBuffer(tiling_context_->GetRawTilingData()->GetData(),
                            tiling_context_->GetRawTilingData()->GetCapacity());
  tiling_context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
  tiling_context_->SetBlockDim(core_num_);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForGroupMatmulFunc(gert::TilingContext *context){
  TilingGroupMatmulFunc tilingObject(context);
  auto ret = tilingObject.Init();
  if(ret != ge::GRAPH_SUCCESS){
    OP_LOGE(context->GetNodeName(), "tiling Init failed.");
    return ge::GRAPH_FAILED;
  }
  ret = tilingObject.RunKernelTiling();
  return ret;
}

} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Shape *x1_shape = context->GetInputShape(0);
  gert::Shape *y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class IndexGroupMatmul : public OpDef {
public:
  explicit IndexGroupMatmul(const char *name) : OpDef(name) {
    this->Input("a")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("b")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8})
        .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});  
    this->Input("scale")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_UINT64, ge::DT_UINT64, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("per_token_scale")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("group_list")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("c")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->SetInferShape(ge::InferShape);

    this->AICore().SetTiling(optiling::TilingForGroupMatmulFunc);
    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(IndexGroupMatmul);
} // namespace ops 
