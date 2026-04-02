/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <register/op_impl_registry.h>
#include "x_flash_attention_infer_tiling.h"
#define ASCENDC_EXTERN_C

namespace optiling {
ge::graphStatus XFAInferTiling::FillBasicTilingData()
{
  auto queryShape = tiling_context_->GetInputShape(InputPosIndex::QUERY)->GetStorageShape();
  auto keyShape = tiling_context_->GetInputShape(InputPosIndex::KEY)->GetStorageShape();
  auto valueShape = tiling_context_->GetInputShape(InputPosIndex::VALUE)->GetStorageShape();
  auto blockTableShape = tiling_context_->GetInputShape(InputPosIndex::BLOCK_TABLE)->GetStorageShape();
  auto actualSeqShape = tiling_context_->GetInputShape(InputPosIndex::Q_LENS)->GetStorageShape();
  auto maskOptionalShapePtr = tiling_context_->GetOptionalInputShape(InputPosIndex::MASK);
  auto attrs = tiling_context_->GetAttrs();
  const int32_t kvHeadNum = *(attrs->GetInt(AttrsIndex::KVHEAD_IDX));
  layout = attrs->GetStr(AttrsIndex::LAYOUT_IDX);
  // TND
  int32_t numTokens = queryShape.GetDim(0);
  int32_t qHeadNum = queryShape.GetDim(1);
  int32_t embeddingSize = queryShape.GetDim(2);
  int32_t batch = actualSeqShape.GetDim(0);
  int32_t maxNumBlocksPerBatch = blockTableShape.GetDim(1);
  float scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * embeddingSize));
  int32_t blockNum, blockSize = 0;
  if (maskOptionalShapePtr != nullptr) {
    maskType = 1;
  }

  if (keyShape.GetDimNum() == NUM4 && valueShape.GetDimNum() == NUM4) { 
    auto lastDimValue = keyShape.GetDim(3);
    if (lastDimValue == NZ_LAST_DIM){ // for NZ
      kvLayout = "NZ";
      blockNum = keyShape.GetDim(0);
      blockSize = keyShape.GetDim(2);
    } else {
      kvLayout = "TND";
      blockNum = keyShape.GetDim(0);
      blockSize = keyShape.GetDim(1);
    }
  } else {
    return ge::GRAPH_FAILED;
  }


  // 设置tiling信息
  tiling_data_.set_batch(batch);
  tiling_data_.set_numTokens(numTokens);
  tiling_data_.set_numHeads(qHeadNum);
  tiling_data_.set_kvHeads(kvHeadNum);
  tiling_data_.set_embeddingSize(embeddingSize);
  tiling_data_.set_embeddingSizeV(embeddingSize);
  tiling_data_.set_scaleValue(scaleValue);
  tiling_data_.set_maskType(maskType);
  tiling_data_.set_numBlocks(blockNum);
  tiling_data_.set_blockSize(blockSize);
  tiling_data_.set_maxNumBlocksPerBatch(maxNumBlocksPerBatch);
  return ge::GRAPH_SUCCESS;
}

uint64_t XFAInferTiling::GetTilingKey() const {

  constexpr uint64_t FLASH_ATTENTION_INFER_BASE_KEY = 1000000000000000000;

  constexpr uint64_t COMP_CAUSAL_MASK_KEY = 3;

  constexpr uint64_t DTYPE_FP16_KEY = 10;
  constexpr uint64_t DTYPE_BF16_KEY = 20;

  constexpr uint64_t KV_LAYOUT_TND_KEY = 100;
  constexpr uint64_t KV_LAYOUT_NZ_KEY = 200;

  constexpr uint64_t FD_KEY = 1000;  // FD 模式的 key 偏移

  uint64_t tilingKey = FLASH_ATTENTION_INFER_BASE_KEY;

  if (kvLayout == "TND") {
    tilingKey += static_cast<uint64_t>(KV_LAYOUT_TND_KEY);
  } else if (kvLayout == "NZ") {
    tilingKey += static_cast<uint64_t>(KV_LAYOUT_NZ_KEY);
  }

  // 0: fp16/half, 1: bf16
  if (inputDtype == 0) {
      tilingKey += static_cast<uint64_t>(DTYPE_FP16_KEY);
  } else if (inputDtype == 1) {
      tilingKey += static_cast<uint64_t>(DTYPE_BF16_KEY);
  }

  if (maskType == 1) {
    tilingKey += static_cast<uint64_t>(COMP_CAUSAL_MASK_KEY);
  }
  
  // 只有满足 FD 条件（maxQSeqlen * numHeads < 128）时才走 FD tiling key
  if (usingFD) {
    tilingKey += static_cast<uint64_t>(FD_KEY);
  }

  return tilingKey;
}

ge::graphStatus XFAInferTiling::ParseInputShapeAndAttrs()
{
  auto dType = tiling_context_->GetInputTensor(InputPosIndex::QUERY)->GetDataType();
  if (dType == ge::DT_FLOAT16) {
      inputDtype = 0;
  } else if (dType == ge::DT_BF16) {
      inputDtype = 1;
  }

  return FillBasicTilingData();
}

void XFAInferTiling::FillSplitCoreTilingDataForJD()
{
  
  uint32_t totalTaskNum = 0;
  uint32_t groupSize = tiling_data_.get_numHeads() / tiling_data_.get_kvHeads();
  // 当前不支持 qSeqlen不等长
  int64_t qSeqlen = tiling_data_.get_numTokens() / tiling_data_.get_batch();
  uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
  uint32_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
  uint32_t curQNBlockNum = qNBlockNumPerGroup * tiling_data_.get_kvHeads();
  uint32_t curQSBlockTile = Q_TILE_CEIL;
  uint32_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
  uint32_t curTaskNum = curQNBlockNum * curQSBlockNum;
  uint32_t firstBatchTaskNum = curTaskNum;
  totalTaskNum = curTaskNum * tiling_data_.get_batch();
  tiling_data_.set_firstBatchTaskNum(firstBatchTaskNum);
  tiling_data_.set_totalTaskNum(totalTaskNum);
}

void XFAInferTiling::SetWorkspaces()
{
  auto platform_info =
      platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
  size_t systemWorkspaceSize = static_cast<size_t>(platform_info.GetLibApiWorkSpaceSize());
  size_t userWorkspaceSize = 0;
  uint64_t mm1OutSize = cubeCoreNum * WORKSPACE_BLOCK_SIZE_DB * NUM4 * NUM3;
  uint64_t smOnlineOutSize = cubeCoreNum * WORKSPACE_BLOCK_SIZE_DB * NUM2 * NUM3;
  uint64_t mm2OutSize = cubeCoreNum * WORKSPACE_BLOCK_SIZE_DB * NUM4 * NUM3;
  uint64_t updateSize = cubeCoreNum * WORKSPACE_BLOCK_SIZE_DB * NUM4 * NUM3;

  uint64_t splitLseTotalSize = cubeCoreNum * tiling_data_.get_numTokens() * tiling_data_.get_numHeads() * sizeof(float);
  uint64_t splitOTotalSize = splitLseTotalSize * tiling_data_.get_embeddingSizeV();
  tiling_data_.set_splitLseTotalSize(splitLseTotalSize);
  tiling_data_.set_splitOTotalSize(splitOTotalSize);


  uint64_t workSpaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + updateSize + splitLseTotalSize + splitOTotalSize;
  tiling_data_.set_mm1OutSize(mm1OutSize);
  tiling_data_.set_smOnlineOutSize(smOnlineOutSize);
  tiling_data_.set_mm2OutSize(mm2OutSize);
  tiling_data_.set_updateSize(updateSize);

  tiling_data_.set_workSpaceSize(workSpaceSize); // new line

  userWorkspaceSize = workSpaceSize;
  size_t* workspace = tiling_context_->GetWorkspaceSizes(1);
  workspace[0] = systemWorkspaceSize + userWorkspaceSize;
}

ge::graphStatus XFAInferTiling::RunTiling()
{
  // Get platform hardware information
  auto platform_info =
      platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
  cubeCoreNum = platform_info.GetCoreNumAic();
  vecCoreNum = platform_info.GetCoreNumAiv();
  blockNum_ = cubeCoreNum;
  auto ret = ParseInputShapeAndAttrs();
  if (ret != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  FillSplitCoreTilingDataForJD();
  SetWorkspaces();

  // Save tilingData
  tiling_data_.SaveToBuffer(tiling_context_->GetRawTilingData()->GetData(),
                            tiling_context_->GetRawTilingData()->GetCapacity());
  tiling_context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
  tiling_context_->SetBlockDim(cubeCoreNum);
  tiling_context_->SetTilingKey(GetTilingKey());
  return ge::GRAPH_SUCCESS;
}


ASCENDC_EXTERN_C ge::graphStatus TilingFunc(gert::TilingContext *context)
{
  XFAInferTiling tilingObject(context);
  auto ret = tilingObject.RunTiling();

  if (ret != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFlashAttentionScore(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    auto compileInfoPtr = context->GetCompiledInfo<FlashAttentionScoreCompileInfo>();

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(XFlashAttentionInfer)
    .Tiling(TilingFunc)
    .TilingParse<FlashAttentionScoreCompileInfo>(TilingPrepareForFlashAttentionScore);  // 向框架注册入口函数;

}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {
class XFlashAttentionInfer : public OpDef {
public:
    explicit XFlashAttentionInfer(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ});
        this->Input("value_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ});
        this->Input("mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actual_q_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actual_kv_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("extra_tiling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("attn_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("layout").String("TND");
        this->Attr("qHead").Int();
        this->Attr("kvHead").Int();
        this->Attr("scale").Float(1.0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        // Configure AI Core support
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(XFlashAttentionInfer);
}  // namespace ops
