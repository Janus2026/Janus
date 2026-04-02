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

#include "beam_search_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define OP_LOGD(nodeName, fmt, ...) \
  printf(fmt, ##__VA_ARGS__);       \
  printf("\n")
#define OP_LOGE(nodeName, fmt, ...) \
  printf(fmt, ##__VA_ARGS__);       \
  printf("\n")

namespace optiling {
class TilingBeamSearchFunc {
 public:
  explicit TilingBeamSearchFunc(gert::TilingContext* tiling_context)
      : tiling_context_(tiling_context) {}

  ge::graphStatus Init();
  ge::graphStatus RunKernelTiling();

 private:
  BeamSearchTilingData tiling_data_;
  gert::TilingContext* tiling_context_ = nullptr;

  void SetTilingKey();
  void FillTilingData();
// TODO: uint32_t to int32_t
  uint32_t num_sequences_ = 0;
  uint32_t block_num_ = 0;
  uint32_t sequence_length_ = 0;
  uint32_t beam_width_ = 0;
  uint32_t top_k_ = 0;
  uint32_t core_num_ = 0;
  uint32_t request_num_ = 0;
  uint32_t min_size_ = 0;
  uint32_t step_size_ = 0;
  size_t sync_workspace_size_ = 0;
};

ge::graphStatus TilingBeamSearchFunc::Init() {
  auto platform_info =
      platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
     uint32_t aiv_num = platform_info.GetCoreNumAiv();
  // uint32_t aiv_num = 1;

  // check input shape
  auto token_ids_shape = tiling_context_->GetInputShape(0)->GetOriginShape();
  auto top_tokens_shape = tiling_context_->GetInputShape(2)->GetOriginShape();

  num_sequences_ = token_ids_shape.GetDim(token_ids_shape.GetDimNum() - 2);
  sequence_length_ = token_ids_shape.GetDim(token_ids_shape.GetDimNum() - 1);
  top_k_ = top_tokens_shape.GetDim(top_tokens_shape.GetDimNum() - 1);
  beam_width_ = top_k_;
  request_num_ = num_sequences_ / beam_width_;
  core_num_ = aiv_num;
  int32_t block_size1 = 32/sizeof(float);
  int32_t align_top_k = (top_k_+block_size1-1)/block_size1*block_size1;
  // int32_t step_size = beam_width_ /48 == 0 ? beam_width_%48 : 48;
  int32_t align_beam_width = (beam_width_+31)/32*32;
  // step_size_ =1024/align_beam_width;
  step_size_ = 8;
  int32_t block_size = (step_size_ * align_top_k + 31)/32*32;
  uint32_t max_size = 0;
  uint32_t min_size0 = 0;
  uint32_t min_size1 = 0;
  uint32_t dtype_size = sizeof(float);
  AscendC::TopKTilingFunc(platform_info,
                          block_size,
                          1,
                          align_top_k,
                          dtype_size,
                          false,
                          AscendC::TopKMode::TOPK_NORMAL,
                          true,
                          tiling_data_.topkTilingData);
  AscendC::GetTopKMaxMinTmpSize(platform_info,
                                block_size,
                                1,
                                false,
                                false,
                                AscendC::TopKMode::TOPK_NORMAL,
                                true,
                                dtype_size,
                                max_size,
                                min_size0);
//   tiling.set_tmpsize(minsize);
int32_t block_size2 = (2 * align_top_k + 31)/32*32;
  AscendC::TopKTilingFunc(platform_info,
                          block_size2, // inner
                          1, // outter
                          align_top_k, // k
                          dtype_size,// dtype
                          true,
                          AscendC::TopKMode::TOPK_NORMAL,
                          true,
                          tiling_data_.topKTilingData1);
  AscendC::GetTopKMaxMinTmpSize(platform_info,
                                block_size2,
                                1,
                                false,
                                true,
                                AscendC::TopKMode::TOPK_NORMAL,
                                true,
                                dtype_size,
                                max_size,
                                min_size1);
  min_size_ = static_cast<int32_t>(std::max(min_size0, min_size1));

  sync_workspace_size_ =
      static_cast<size_t>(platform_info.GetLibApiWorkSpaceSize());
  return ge::GRAPH_SUCCESS;
}

void TilingBeamSearchFunc::SetTilingKey() { tiling_context_->SetTilingKey(0); }

void TilingBeamSearchFunc::FillTilingData() {
  tiling_data_.set_num_sequences(num_sequences_);
  tiling_data_.set_sequence_length(sequence_length_);
  tiling_data_.set_beam_width(beam_width_);
  tiling_data_.set_top_k(top_k_);
  tiling_data_.set_request_num(request_num_);
  tiling_data_.set_core_num(core_num_);
  tiling_data_.set_min_size(min_size_);
  tiling_data_.set_step_size(step_size_);
}

ge::graphStatus TilingBeamSearchFunc::RunKernelTiling() {
  SetTilingKey();
  FillTilingData();
  size_t userWorkspaceSize = 0;
  size_t* currentWorkspace = tiling_context_->GetWorkspaceSizes(1);
  currentWorkspace[0] = userWorkspaceSize + sync_workspace_size_;
  tiling_data_.SaveToBuffer(tiling_context_->GetRawTilingData()->GetData(),
                            tiling_context_->GetRawTilingData()->GetCapacity());
  tiling_context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
  tiling_context_->SetBlockDim(core_num_);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForBeamSearchFunc(gert::TilingContext* context) {
  TilingBeamSearchFunc tilingObject(context);
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
class BeamSearch : public OpDef {
 public:
  explicit BeamSearch(const char* name) : OpDef(name) {
    this->Input("log_probs")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("top_tokens")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("top_probs")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("out_token_ids")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("out_token_index")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("out_log_probs")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->SetInferShape(ge::InferShape);
    this->AICore().SetTiling(optiling::TilingForBeamSearchFunc);
    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(BeamSearch);
}  // namespace ops
