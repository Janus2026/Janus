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

#include "x_attention_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>

#define OP_LOGD(nodeName, fmt, ...) \
  printf(fmt, ##__VA_ARGS__);       \
  printf("\n")
#define OP_LOGE(nodeName, fmt, ...) \
  printf(fmt, ##__VA_ARGS__);       \
  printf("\n")


namespace optiling {

enum InputIndex {
    QUERY = 0,
    SHARED_KEY_BLOCK,
    SHARED_VALUE_BLOCK,
    UNSHARED_KEY_BLOCK,
    UNSHARED_VALUE_BLOCK,
    UNSHARED_BLOCK_TABLE,
    SHARED_KV_LENS,
    DECODE_STEP,
    SHARED_BLOCK_TABLE,
};

constexpr int32_t NUM2 = 2;
constexpr int32_t NUM3 = 3;
constexpr int32_t NUM4 = 4;
constexpr int32_t UNSHARED_Q_TILE = 128;
constexpr int32_t UNSHARED_KV_TILE = 256;
constexpr uint32_t Q_S_BLOCK_TILE = 128;
constexpr uint32_t BLOCK_SIZE = 128;
constexpr int32_t WORKSPACE_BLOCK_SIZE_DB = 128 * 128 * 4; // row * col * blockStackNum
constexpr int32_t UNSHARED_WORKSPACE_BLOCK_SIZE_DB = 128 * 256;   // unshared no pinpong
constexpr int32_t FLOAT_BLOCK_SIZE = 8;

class TilingXAttentionFunc {
  public:
    explicit TilingXAttentionFunc(gert::TilingContext* tiling_context)
        : tiling_context_(tiling_context) {}
    ge::graphStatus RunTiling();
  private:
    uint64_t GetTilingKey() const;
  private:
    XAttentionTilingData tiling_data_;
    gert::TilingContext* tiling_context_ = nullptr;
    uint32_t sharedBlockDim = 0;
    uint32_t unsharedBlockDim = 0;
    uint32_t cubeCoreNum;
    uint32_t vecCoreNum;
    bool isSharedPaged{false};
    bool isUnsharedPaged{false};
    uint32_t inputDtype{0};
    ge::graphStatus ParseInputShapeAndAttrs();
    ge::graphStatus FillBasicTilingData();
    ge::graphStatus FillBasicTilingData4NewKind();
    void FillSharedSplitCoreTilingData();
    void FillUnsharedSplitCoreTilingData();
    void FillCombineScaleTilingData();
    void BalanceAicore();
    void SetWorkspaces();
    uint32_t GetQNBlockTile(int64_t qSeqlen, uint32_t groupSize);
    
};


ge::graphStatus TilingXAttentionFunc::FillBasicTilingData4NewKind()
{
  auto queryShape = tiling_context_->GetInputShape(QUERY)->GetStorageShape();
  // shared [total_num_tokens, kv_head, head_dim]
  auto sharedKeyBlockShape = tiling_context_->GetInputShape(SHARED_KEY_BLOCK)->GetStorageShape();
  // unshared [max_request_num, beamsize, kv_head, max_decode_step, head_dim]
  // unshared_blk_tb [bs, request_idx]
  auto unsharedKeyBlockShape = tiling_context_->GetInputShape(UNSHARED_KEY_BLOCK)->GetStorageShape();
  auto unsharedBlockTableShape = tiling_context_->GetOptionalInputShape(UNSHARED_BLOCK_TABLE)->GetStorageShape();
  
  int32_t numTokens = queryShape.GetDim(0);
  int32_t qHeadNum = queryShape.GetDim(1);
  int32_t embeddingSize = queryShape.GetDim(2);
  int32_t batch = unsharedBlockTableShape.GetDim(0);
  int32_t kvHeadNum = sharedKeyBlockShape.GetDim(1);
  int32_t maxDecodeStep = unsharedKeyBlockShape.GetDim(NUM3);
  int32_t beamSize = numTokens / batch;

  float scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * embeddingSize));

  // set tiling data
  tiling_data_.set_batch(batch);
  tiling_data_.set_numHeads(qHeadNum);
  tiling_data_.set_kvHeads(kvHeadNum);
  tiling_data_.set_embeddingSize(embeddingSize);
  tiling_data_.set_beamSize(beamSize);
  tiling_data_.set_scaleValue(scaleValue);
  tiling_data_.set_maskType(0);
  tiling_data_.set_blockSize(BLOCK_SIZE);
  tiling_data_.set_numTokens(numTokens);
  tiling_data_.set_maxDecodeStep(maxDecodeStep);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingXAttentionFunc::ParseInputShapeAndAttrs()
{
    auto dType = tiling_context_->GetInputTensor(InputIndex::QUERY)->GetDataType();
    if (dType == ge::DT_FLOAT16) {
        inputDtype = 0;
    } else if (dType == ge::DT_BF16) {
        inputDtype = 1;
    }
    auto sharedBlockTableShapePtr = tiling_context_->GetOptionalInputShape(InputIndex::SHARED_BLOCK_TABLE);
    auto unsharedBlockTableShapePtr = tiling_context_->GetOptionalInputShape(InputIndex::UNSHARED_BLOCK_TABLE);
    isSharedPaged = (sharedBlockTableShapePtr != nullptr);
    isUnsharedPaged = (unsharedBlockTableShapePtr != nullptr);
    if (isSharedPaged && !isUnsharedPaged) {
        return FillBasicTilingData();
    } else if (!isSharedPaged && isUnsharedPaged) {
        return FillBasicTilingData4NewKind();
    }
    OP_LOGE(tiling_context_->GetNodeName(), "unexpected input combination between shared_block_table and unshared_block_table.");
    return ge::GRAPH_FAILED;
}

void TilingXAttentionFunc::BalanceAicore()
{
  // Support dynamic calculation based on the amount of computation in the future.
  sharedBlockDim = 12;
  unsharedBlockDim = cubeCoreNum - sharedBlockDim;
  return;
}

uint32_t TilingXAttentionFunc::GetQNBlockTile(int64_t qSeqlen, uint32_t groupSize)
{
    uint32_t qRowNumCeil = 128;
    // A trick is used to ensure the qN tile is a even number,
    // thus most tasks have balanced workload between two vec cores,
    // and each vec core possess no more than 64 rows when all-rounded row num is no larger than 128,
    // aiding the coding of rescale block
    uint32_t qNBlockTile = (qRowNumCeil / qSeqlen) / 2 * 2;
    qNBlockTile = std::min(qNBlockTile, groupSize);
    qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
    return qNBlockTile;
}


void TilingXAttentionFunc::FillUnsharedSplitCoreTilingData()
{
  tiling_data_.set_unsharedCoreNum(unsharedBlockDim);
  tiling_data_.set_groupSize(tiling_data_.get_numHeads() / tiling_data_.get_kvHeads());
  uint32_t totalGroupCount = tiling_data_.get_beamSize() * tiling_data_.get_kvHeads();
  // for no PA scenario, calculation can cross batch
  if (!isUnsharedPaged) {
    totalGroupCount *= tiling_data_.get_batch();
  }
  uint32_t maxGroupCountPerLoop = std::min(UNSHARED_Q_TILE / tiling_data_.get_groupSize(),
                                           UNSHARED_KV_TILE / tiling_data_.get_maxDecodeStep());
  // ensure each task handles same group count
  while (maxGroupCountPerLoop > 1 &&
         (totalGroupCount % maxGroupCountPerLoop != 0 || maxGroupCountPerLoop % FLOAT_BLOCK_SIZE != 0))
      --maxGroupCountPerLoop;
  tiling_data_.set_unshareGroupCountPerLoop(maxGroupCountPerLoop);
  uint32_t totalTaskNum = totalGroupCount / maxGroupCountPerLoop;
  if (isUnsharedPaged) {
    tiling_data_.set_unsharedLoopCountPerBatch(totalTaskNum);
    totalTaskNum *= tiling_data_.get_batch();
  }
  uint32_t unsharedFullCoreNum = unsharedBlockDim;
  uint32_t unsharedTaskNumHead = totalTaskNum / unsharedBlockDim;
  uint32_t unsharedTaskNumTail = unsharedTaskNumHead;
  uint32_t remainTask = totalTaskNum % unsharedBlockDim;
  if (remainTask != 0) {
    unsharedFullCoreNum = remainTask;
    unsharedTaskNumHead += 1;
  }
  tiling_data_.set_unsharedFullCoreNum(unsharedFullCoreNum);
  tiling_data_.set_unsharedTaskNumHead(unsharedTaskNumHead);
  tiling_data_.set_unsharedTaskNumTail(unsharedTaskNumTail);

}


void TilingXAttentionFunc::SetWorkspaces()
{
  auto platform_info =
      platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
  size_t systemWorkspaceSize = static_cast<size_t>(platform_info.GetLibApiWorkSpaceSize());
  size_t userWorkspaceSize = 0;
  
  uint64_t qoSize = tiling_data_.get_numTokens() 
                    * tiling_data_.get_numHeads()
                    * tiling_data_.get_embeddingSize()
                    * sizeof(int16_t);
  // Attention occupied space
  // TODO: Only apply for one temporary space, affecting preload function, long sequence scenario needs extra processing
  uint64_t mm1OutSize = (sharedBlockDim * WORKSPACE_BLOCK_SIZE_DB + 
                         unsharedBlockDim * UNSHARED_WORKSPACE_BLOCK_SIZE_DB) * NUM3 * sizeof(float);;
  uint64_t smOnlineOutSize = (sharedBlockDim * WORKSPACE_BLOCK_SIZE_DB +
                              unsharedBlockDim * UNSHARED_WORKSPACE_BLOCK_SIZE_DB) * NUM3 * sizeof(int16_t);
  uint64_t mm2OutSize = sharedBlockDim * WORKSPACE_BLOCK_SIZE_DB * NUM3 * sizeof(float);
  // oUpdate currently not used
  uint64_t updateSize = 0; // sharedBlockDim * WORKSPACE_BLOCK_SIZE_DB * NUM3 * sizeof(float);
  tiling_data_.set_mm1OutSize(mm1OutSize);
  tiling_data_.set_smOnlineOutSize(smOnlineOutSize);
  tiling_data_.set_mm2OutSize(mm2OutSize);
  tiling_data_.set_updateSize(updateSize);

  // combine required output occupied space
  uint64_t sumMaxSize = tiling_data_.get_numTokens() * tiling_data_.get_numHeads() * sizeof(float) * NUM2;
  uint64_t attnOutSize = qoSize * 2;
  uint64_t combineWorkspaceSize = sumMaxSize * FLOAT_BLOCK_SIZE + attnOutSize;
  uint64_t unsharedcombineWorkspaceSize = sumMaxSize + attnOutSize;
  tiling_data_.set_sharedWorkspaceSize(combineWorkspaceSize); // new line

  userWorkspaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + updateSize + combineWorkspaceSize +
                      unsharedcombineWorkspaceSize;
  size_t* workspace = tiling_context_->GetWorkspaceSizes(1);
  workspace[0] = systemWorkspaceSize + userWorkspaceSize;
}

void TilingXAttentionFunc::FillCombineScaleTilingData()
{
  uint32_t rowNum = tiling_data_.get_batch() * 
                    tiling_data_.get_beamSize() * 
                    tiling_data_.get_numHeads();
  uint32_t columnSize = tiling_data_.get_embeddingSize();
  
  uint32_t rowNumPerCore = rowNum / cubeCoreNum;  // number of rows per core
  uint32_t rowNumTailPerCore = rowNum % cubeCoreNum;  // remaining rows, need to be allocated to the first few cores
  tiling_data_.set_combineFormerCoreNum(rowNumTailPerCore);
  tiling_data_.set_combineFormerRowNum(rowNumPerCore + 1);
  tiling_data_.set_combineTailRowNum(rowNumPerCore);
  tiling_data_.set_combineCoreNum(cubeCoreNum);
}

void TilingXAttentionFunc::FillSharedSplitCoreTilingData()
{
  uint32_t totalTaskNum = 0;
  uint32_t groupSize = tiling_data_.get_numHeads() / tiling_data_.get_kvHeads();
  int64_t qSeqlen = tiling_data_.get_beamSize();
  uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
  uint32_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
  uint32_t curQNBlockNum = qNBlockNumPerGroup * tiling_data_.get_kvHeads();
  uint32_t curQSBlockTile = Q_S_BLOCK_TILE;
  uint32_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
  uint32_t curTaskNum = curQNBlockNum * curQSBlockNum;
  uint32_t firstSharedBatchTaskNum = curTaskNum;
  totalTaskNum = curTaskNum * tiling_data_.get_batch();
  tiling_data_.set_firstSharedBatchTaskNum(firstSharedBatchTaskNum);
  tiling_data_.set_sharedTotalTaskNum(totalTaskNum);
  tiling_data_.set_sharedCoreNum(sharedBlockDim);
}


ge::graphStatus TilingXAttentionFunc::FillBasicTilingData()
{
  auto queryShape = tiling_context_->GetInputShape(QUERY)->GetStorageShape();
  auto sharedKeyBlockShape = tiling_context_->GetInputShape(SHARED_KEY_BLOCK)->GetStorageShape();
  auto unsharedKeyBlockShape = tiling_context_->GetInputShape(UNSHARED_KEY_BLOCK)->GetStorageShape();
  auto sharedBlockTableShape = tiling_context_->GetOptionalInputShape(SHARED_BLOCK_TABLE)->GetStorageShape();
  
  int32_t numTokens = queryShape.GetDim(0);
  int32_t qHeadNum = queryShape.GetDim(1);
  int32_t embeddingSize = queryShape.GetDim(2);
  int32_t batch = sharedBlockTableShape.GetDim(0);
  int32_t maxNumBlocksPerBatch = sharedBlockTableShape.GetDim(1);
  int32_t blockNum = sharedKeyBlockShape.GetDim(0);
  int32_t blockSize = sharedKeyBlockShape.GetDim(1);
  int32_t kvHeadNum = sharedKeyBlockShape.GetDim(2);
  int32_t maxDecodeStep = unsharedKeyBlockShape.GetDim(2);
  int32_t beamSize = numTokens / batch;

  float scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * embeddingSize));

  // 设置tiling信息
  tiling_data_.set_batch(batch);
  tiling_data_.set_numHeads(qHeadNum);
  tiling_data_.set_kvHeads(kvHeadNum);
  tiling_data_.set_embeddingSize(embeddingSize);
  tiling_data_.set_beamSize(beamSize);
  tiling_data_.set_scaleValue(scaleValue);
  tiling_data_.set_maskType(0);
  tiling_data_.set_numTokens(numTokens);
  tiling_data_.set_numBlocks(blockNum);
  tiling_data_.set_blockSize(blockSize);
  tiling_data_.set_maxNumBlocksPerBatch(maxNumBlocksPerBatch);
  tiling_data_.set_maxDecodeStep(maxDecodeStep);
  return ge::GRAPH_SUCCESS;
}

uint64_t TilingXAttentionFunc::GetTilingKey() const {
    uint64_t resKey = 0;
    resKey = uint32_t(isSharedPaged << NUM3) + uint32_t(isUnsharedPaged << NUM2) + (inputDtype << 1);
    // shared continous unshared paged  key: bf16(6) fp16(4) 0 1 (0/1)
    // shared paged unshared continous key: bf16(10) fp16(8) 1 0 (0/1)
    return resKey;
}

ge::graphStatus TilingXAttentionFunc::RunTiling()
{
  // Get platform hardware information
  auto platform_info =
      platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
  cubeCoreNum = platform_info.GetCoreNumAic();
  vecCoreNum = platform_info.GetCoreNumAiv();

  BalanceAicore();
  auto ret = ParseInputShapeAndAttrs();
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(tiling_context_->GetNodeName(), "fill basic tiling failed.");
    return ge::GRAPH_FAILED;
  }

  FillSharedSplitCoreTilingData();
  FillUnsharedSplitCoreTilingData();
  FillCombineScaleTilingData();
  SetWorkspaces();

  // Save tilingData
  tiling_data_.SaveToBuffer(tiling_context_->GetRawTilingData()->GetData(),
                            tiling_context_->GetRawTilingData()->GetCapacity());
  tiling_context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
  tiling_context_->SetBlockDim(cubeCoreNum);

  tiling_context_->SetTilingKey(GetTilingKey());

  return ge::GRAPH_SUCCESS;
}


static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  TilingXAttentionFunc tilingObject(context);
  auto ret = tilingObject.RunTiling();

  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(context->GetNodeName(), "xAttention tiling failed.");
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
  const gert::Shape* x1_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class XAttention : public OpDef {
public:
    explicit XAttention(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("shared_key_block")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("shared_value_block")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("unshared_key_block")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("unshared_value_block")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("unshared_block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("shared_kv_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("decode_step")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("shared_block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("attn_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

    }
};

OP_ADD(XAttention);
}
