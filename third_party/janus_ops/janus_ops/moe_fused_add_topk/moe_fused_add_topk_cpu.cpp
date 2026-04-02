#include "register/op_def_registry.h"
// #include "platform/platform_info.h"
// #include "runtime2_util.h"
// #include "op_log.h"
#include "tiling/tiling_api.h"
// #include "tiling/tiling_type.h"
#include "moe_fused_add_topk_tiling.h"

#define OP_LOGD(nodeName, fmt, ...) printf(fmt, ##__VA_ARGS__); printf("\n")
#define OP_LOGE(nodeName, fmt, ...) printf(fmt, ##__VA_ARGS__); printf("\n")

namespace {
    constexpr uint32_t BYTE_BLOCK = 32;
    constexpr int32_t X_INPUT_INDEX = 0;
    constexpr int32_t ADD_NUM_INPUT_INDEX = 1;
    constexpr int32_t MAPPING_NUM_INPUT_INDEX = 2;
    constexpr int32_t MAPPING_TABLE_INPUT_INDEX = 3;

    constexpr uint32_t DIM_INDEX0 = 0;
    constexpr uint32_t DIM_INDEX1 = 1;

    constexpr uint32_t ATTR_GROUP_NUM_INDEX = 0;
    constexpr uint32_t ATTR_GROUP_TOPK_INDEX = 1;
    constexpr uint32_t ATTR_TOP_N_INDEX = 2;
    constexpr uint32_t ATTR_TOP_K_INDEX = 3;
    constexpr uint32_t ATTR_ACTIVATE_TYPE_INDEX = 4;
    constexpr uint32_t ATTR_IS_NORM_INDEX = 5;
    constexpr uint32_t ATTR_SCALE_INDEX = 6;
    constexpr uint32_t ATTR_ENABLE_EXPERT_MAPPING_INDEX = 7;

    constexpr uint32_t RESERVED_UB = 16 * 1024;
    constexpr int32_t NUM_FOUR = 4;
    constexpr int32_t SORT_UNIT = 32;
    constexpr int32_t GROUP_LIMIT = 8;
    constexpr int32_t MAPPING_TABLE_DIM_ONE_LIMIT = 128;
    constexpr int32_t NUM_SIXTEEN = 16;
    constexpr int32_t NUM_TEN = 10;
    constexpr uint32_t SYS_WORKSPACESIZE = 16 * 1024 * 1024;

    constexpr bool TOPK_IS_REUSE_SOURCE = false;
    constexpr bool TOPK_IS_INIT_INDEX = false;
    constexpr bool TOPK_IS_LARGEST = true;
    constexpr uint32_t FP32_DTYPE_SIZE = 4U; // float
    constexpr uint32_t FP32_BLOCK_ALIGN_NUM = 8U;
} // namespace

namespace optiling {

class MoeFusedAddTopkTiling {
public:
    explicit MoeFusedAddTopkTiling(gert::TilingContext* context) : tilingContext_(context) {}
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint();

protected:
    void GetTilingKey();
    void GetUsedCore();
    ge::graphStatus SplitUb();
    void FillTilingData();
    ge::graphStatus GetTopKTiling();
    void GetTmpBuffSize();
    int64_t CeilAlign(int64_t u_value, int64_t d_value);

    uint32_t firstDimSize_ = 0;
    uint32_t secondDimSize_ = 0;
    uint32_t addNumDimSize_ = 0;
    uint32_t groupNum_ = 0;
    uint32_t groupTopk_;
    uint32_t topN_;
    uint32_t topK_ = 0;
    uint32_t activateType_ = 0;
    uint32_t isNorm_ = 0;
    float scale_ = 1.0;
    bool enableExpertMapping_ = false;
    uint32_t groupEles_ = 0;
    uint32_t expertNum_ = 0;
    uint32_t tableDim_ = 0;
    uint32_t ubSize_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t batchPerCore_ = 1;
    uint32_t tailBatch_ = 0;
    uint32_t ubFactorElement_ = 0;
    uint32_t topkMaxValue_ = 0;
    uint32_t topkMinValue_ = 0;
    uint64_t tilingKey_ = 0;
    uint64_t workspacePerCore_ = 0;
    MoeFusedAddTopkTilingData tilingData_;
    gert::TilingContext* tilingContext_ = nullptr;
};

ge::graphStatus MoeFusedAddTopkTiling::Init() 
{
    auto xShape = tilingContext_->GetInputShape(X_INPUT_INDEX)->GetStorageShape();
    auto addNumShape = tilingContext_->GetInputShape(ADD_NUM_INPUT_INDEX)->GetStorageShape();
    firstDimSize_ = xShape.GetDim(DIM_INDEX0);
    secondDimSize_ = xShape.GetDim(DIM_INDEX1);
    addNumDimSize_ = addNumShape.GetDim(DIM_INDEX0);

    auto attrs = tilingContext_->GetAttrs();
    groupNum_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_GROUP_NUM_INDEX));
    groupTopk_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_GROUP_TOPK_INDEX));
    topN_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_TOP_N_INDEX));
    topK_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_TOP_K_INDEX));
    activateType_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_ACTIVATE_TYPE_INDEX));
    isNorm_ = static_cast<uint32_t>(*(attrs->GetAttrPointer<bool>(ATTR_IS_NORM_INDEX)));
    scale_ = *(attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX));
    enableExpertMapping_ = *(attrs->GetAttrPointer<bool>(ATTR_ENABLE_EXPERT_MAPPING_INDEX));
    groupEles_ = groupNum_ == 0 ? secondDimSize_ : secondDimSize_ / groupNum_;

    if (enableExpertMapping_) {
        auto mappingTableShape = tilingContext_->GetInputShape(MAPPING_TABLE_INPUT_INDEX)->GetStorageShape();
        expertNum_ = mappingTableShape.GetDim(DIM_INDEX0);
        tableDim_ = mappingTableShape.GetDim(DIM_INDEX1);
    }
    auto platform_info = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    uint32_t aiv_num = platform_info.GetCoreNumAiv();
    uint64_t platformUbSize = 0;
    platform_info.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformUbSize);
    // OPS_CHECK_NULL_WITH_CONTEXT(tilingContext_, compileInfo);
    coreNum_ = aiv_num;
    ubSize_ = (platformUbSize - RESERVED_UB) / BYTE_BLOCK * BYTE_BLOCK;

    GetTilingKey();
    GetUsedCore();
    if (GetTopKTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(tilingContext_->GetNodeName(), "GetTopKTiling Failed");
        return ge::GRAPH_FAILED;
    }
    GetTmpBuffSize();
    return SplitUb();
}

void MoeFusedAddTopkTiling::GetTilingKey()
{
    tilingKey_ = 0U;
    if (enableExpertMapping_) {
        tilingKey_ += 1U;
    }
}

void MoeFusedAddTopkTiling::GetUsedCore()
{
    if (firstDimSize_ <= coreNum_) {
        batchPerCore_ = 1;
        usedCoreNum_ = firstDimSize_;
        tailBatch_ = 0;
        return;
    }
    batchPerCore_ = firstDimSize_ / coreNum_;
    tailBatch_ = firstDimSize_ % coreNum_;
    usedCoreNum_ = coreNum_;
    return;
}

int64_t MoeFusedAddTopkTiling::CeilAlign(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }
  res_value = (u_value + d_value - 1) / d_value * d_value;

  return res_value;
}

ge::graphStatus MoeFusedAddTopkTiling::SplitUb()
{
    uint32_t needUbSize = 0;
    uint32_t tilingDataSize = (sizeof(tilingData_) + BYTE_BLOCK - 1) / BYTE_BLOCK * BYTE_BLOCK;
    ubFactorElement_ = (ubSize_ - tilingDataSize) / BYTE_BLOCK;

    // tilingDataSize
    needUbSize += tilingDataSize;
    // xInQueue_
    needUbSize += CeilAlign(groupEles_, SORT_UNIT) * groupNum_ * FP32_DTYPE_SIZE;
    // addNumInQueue_
    needUbSize += CeilAlign(secondDimSize_, FP32_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    // yOutQueue_
    needUbSize +=  CeilAlign(topK_, FP32_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    // indicesOutQueue_
    needUbSize +=  CeilAlign(topK_, SORT_UNIT) * sizeof(int32_t);
    // sigmoidBuf_
    needUbSize +=  CeilAlign(secondDimSize_, FP32_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    // sigmoidAddQueue_
    needUbSize +=  CeilAlign(secondDimSize_, FP32_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    // sortedQueue_
    needUbSize +=  CeilAlign(groupEles_, SORT_UNIT) * groupNum_ * sizeof(int64_t);
    // topkValueQueue_
    needUbSize +=  CeilAlign(secondDimSize_, SORT_UNIT) * FP32_DTYPE_SIZE;
    // assistQueue_
    needUbSize +=  CeilAlign(secondDimSize_, FP32_BLOCK_ALIGN_NUM) * sizeof(uint32_t);
    // mappingNumQueue_
    needUbSize +=  CeilAlign(expertNum_ * sizeof(int32_t), BYTE_BLOCK);
    // tempBuf_
    needUbSize +=  topkMinValue_;

    if(needUbSize > ubSize_){
        // OP_LOGD(tilingContext_->GetNodeName(), "This case need minimum UB size is %u, which is out of total UB size: %u.",needUbSize, ubSize_);
        return ge::GRAPH_FAILED;
    }
    // OP_TILING_CHECK(needUbSize > ubSize_,
    //         VECTOR_INNER_ERR_REPORT_TILIING(tilingContext_->GetNodeName(),
    //                                         "This case need minimum UB size is %u, which is out of total UB size: %u.",
    //                                         needUbSize, ubSize_),
    //         return ge::GRAPH_FAILED);
    // OP_LOGD(tilingContext_->GetNodeName(), "This case need minimum UB size is %u.", needUbSize);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFusedAddTopkTiling::SetKernelTiling()
{
    workspacePerCore_ = secondDimSize_ * FP32_DTYPE_SIZE;
    size_t usedWorkspaceSize = SYS_WORKSPACESIZE + usedCoreNum_ * workspacePerCore_;

    tilingContext_->SetTilingKey(tilingKey_);
    tilingContext_->SetBlockDim(usedCoreNum_);
    tilingData_.set_firstDimSize(firstDimSize_);
    tilingData_.set_secondDimSize(secondDimSize_);
    tilingData_.set_addNumDimSize(addNumDimSize_);
    tilingData_.set_groupNum(groupNum_);
    tilingData_.set_groupTopk(groupTopk_);
    tilingData_.set_topN(topN_);
    tilingData_.set_topK(topK_);

    tilingData_.set_activateType(activateType_);
    tilingData_.set_isNorm(isNorm_);
    tilingData_.set_scale(scale_);
    tilingData_.set_groupEles(groupEles_);
    tilingData_.set_blockNum(usedCoreNum_);
    tilingData_.set_ubFactorElement(ubFactorElement_);
    tilingData_.set_batchPerCore(batchPerCore_);
    tilingData_.set_tailBatch(tailBatch_);

    tilingData_.set_expertNum(expertNum_);
    tilingData_.set_tableDim(tableDim_);
    tilingData_.set_topkMaxValue(topkMaxValue_);
    tilingData_.set_topkMinValue(topkMinValue_);
    tilingData_.set_workspacePerCore(workspacePerCore_);

    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    workspaces[0] = usedWorkspaceSize;
    
    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    // TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void MoeFusedAddTopkTiling::TilingDataPrint()
{
    OP_LOGD(tilingContext_->GetNodeName(), "tilingKey:                      %lu", tilingKey_);
    OP_LOGD(tilingContext_->GetNodeName(), "usedCoreNum:                    %u", usedCoreNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "firstDimSize:                   %u", tilingData_.get_firstDimSize());
    OP_LOGD(tilingContext_->GetNodeName(), "secondDimSize:                  %u", tilingData_.get_secondDimSize());
    OP_LOGD(tilingContext_->GetNodeName(), "addNumDimSize:                  %u", tilingData_.get_addNumDimSize());
    OP_LOGD(tilingContext_->GetNodeName(), "groupNum:                       %u", tilingData_.get_groupNum());
    OP_LOGD(tilingContext_->GetNodeName(), "groupTopk:                      %u", tilingData_.get_groupTopk());
    OP_LOGD(tilingContext_->GetNodeName(), "topN:                           %u", tilingData_.get_topN());
    OP_LOGD(tilingContext_->GetNodeName(), "topK:                           %u", tilingData_.get_topK());
    OP_LOGD(tilingContext_->GetNodeName(), "activateType:                   %u", tilingData_.get_activateType());
    OP_LOGD(tilingContext_->GetNodeName(), "isNorm:                         %u", tilingData_.get_isNorm());
    OP_LOGD(tilingContext_->GetNodeName(), "scale:                          %f", tilingData_.get_scale());
    OP_LOGD(tilingContext_->GetNodeName(), "groupEles:                      %u", tilingData_.get_groupEles());
    OP_LOGD(tilingContext_->GetNodeName(), "blockNum:                       %u", tilingData_.get_blockNum());
    OP_LOGD(tilingContext_->GetNodeName(), "ubFactorElement:                %u", tilingData_.get_ubFactorElement());
    OP_LOGD(tilingContext_->GetNodeName(), "batchPerCore:                   %u", tilingData_.get_batchPerCore());
    OP_LOGD(tilingContext_->GetNodeName(), "tailBatch:                      %u", tilingData_.get_tailBatch());
    OP_LOGD(tilingContext_->GetNodeName(), "expertNum:                      %u", tilingData_.get_expertNum());
    OP_LOGD(tilingContext_->GetNodeName(), "tableDim:                       %u", tilingData_.get_tableDim());
    OP_LOGD(tilingContext_->GetNodeName(), "topkMaxValue:                   %u", tilingData_.get_topkMaxValue());
    OP_LOGD(tilingContext_->GetNodeName(), "topkMinValue:                   %u", tilingData_.get_topkMinValue());
    OP_LOGD(tilingContext_->GetNodeName(), "workspacePerCore:               %lu", tilingData_.get_workspacePerCore());
}

ge::graphStatus MoeFusedAddTopkTiling::GetTopKTiling()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());

    int32_t topkInner = (groupEles_ + SORT_UNIT - 1) / SORT_UNIT * SORT_UNIT;

    if(!AscendC::TopKTilingFunc(ascendcPlatform, topkInner, groupNum_, topN_, FP32_DTYPE_SIZE, TOPK_IS_INIT_INDEX,
                        AscendC::TopKMode::TOPK_NORMAL, TOPK_IS_LARGEST, tilingData_.topkTilingData))
    {
        // OP_LOGD(tilingContext_->GetNodeName(), "TopKTilingFunc Failed");
        return ge::GRAPH_FAILED; 
    }

    if(!AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, topkInner, groupNum_, TOPK_IS_REUSE_SOURCE, 
                        TOPK_IS_INIT_INDEX, AscendC::TopKMode::TOPK_NORMAL, TOPK_IS_LARGEST, FP32_DTYPE_SIZE,
                        topkMaxValue_, topkMinValue_)){
        // OP_LOGD(tilingContext_->GetNodeName(), "GetTopKMaxMinTmpSize Failed");
        return ge::GRAPH_FAILED; 
    }
    // OP_TILING_CHECK(!AscendC::TopKTilingFunc(ascendcPlatform, topkInner, groupNum_, topN_, FP32_DTYPE_SIZE, TOPK_IS_INIT_INDEX,
    //                     AscendC::TopKMode::TOPK_NORMAL, TOPK_IS_LARGEST, tilingData_.topkTilingData),
    //                 VECTOR_INNER_ERR_REPORT_TILIING(tilingContext_->GetNodeName(),
    //                                                 "TopKTilingFunc Failed"),
    //                 return ge::GRAPH_FAILED);
    // OP_TILING_CHECK(!AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, topkInner, groupNum_, TOPK_IS_REUSE_SOURCE, 
    //                     TOPK_IS_INIT_INDEX, AscendC::TopKMode::TOPK_NORMAL, TOPK_IS_LARGEST, FP32_DTYPE_SIZE,
    //                     topkMaxValue_, topkMinValue_),
    //                 VECTOR_INNER_ERR_REPORT_TILIING(tilingContext_->GetNodeName(),
    //                                                 "GetTopKMaxMinTmpSize Failed"),
    //                 return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

void MoeFusedAddTopkTiling::GetTmpBuffSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    uint32_t groupElesAlignSortCount = (groupNum_ + SORT_UNIT - 1) / SORT_UNIT * SORT_UNIT;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    std::vector<int64_t> sigmoidVec = {static_cast<int64_t>(secondDimSize_)};
    ge::Shape sigmoidShape(sigmoidVec);
    AscendC::GetSigmoidMaxMinTmpSize(sigmoidShape, FP32_DTYPE_SIZE, false, maxValue, minValue);
    topkMinValue_ = std::max(topkMinValue_, minValue);
    topkMaxValue_ = std::max(topkMaxValue_, maxValue);

    uint32_t sortTmpSize = AscendC::GetSortTmpSize(ascendcPlatform, groupElesAlignSortCount, FP32_DTYPE_SIZE);
    topkMinValue_ = std::max(topkMinValue_, sortTmpSize);

    std::vector<int64_t> srcBroadCastVec = {static_cast<int64_t>(groupNum_), 1};
    std::vector<int64_t> dstBroadCastVec = {static_cast<int64_t>(groupNum_), static_cast<int64_t>(groupEles_)};
    ge::Shape srcBroadCastShape(srcBroadCastVec);
    ge::Shape dstBroadCastShape(dstBroadCastVec);
    AscendC::GetBroadCastMaxMinTmpSize(ascendcPlatform, srcBroadCastShape, dstBroadCastShape,
                                        FP32_DTYPE_SIZE, false, maxValue, minValue);
    topkMinValue_ = std::max(topkMinValue_, minValue);
    topkMaxValue_ = std::max(topkMaxValue_, maxValue);

    uint32_t secondDimSizeAlignSortCount = (secondDimSize_ + SORT_UNIT - 1) / SORT_UNIT * SORT_UNIT;
    sortTmpSize = AscendC::GetSortTmpSize(ascendcPlatform, secondDimSizeAlignSortCount, FP32_DTYPE_SIZE);
    topkMinValue_ = std::max(topkMinValue_, sortTmpSize);

    topkMinValue_ = std::max(topkMinValue_, secondDimSizeAlignSortCount * FP32_DTYPE_SIZE);
    topkMaxValue_ = std::max(topkMaxValue_, topkMinValue_);
}



static ge::graphStatus TilingMoeFusedAddTopk(gert::TilingContext *context) {
    // OP_TILING_CHECK(context == nullptr, VECTOR_INNER_ERR_REPORT_TILIING("MoeFusedAddTopk", "context is null"),
    //     return ge::GRAPH_FAILED);
    auto nodeName = context->GetNodeName();
    // OP_LOGD(nodeName, "Tiling initing");
    MoeFusedAddTopkTiling tilingObj(context);
    if (tilingObj.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(nodeName, "tiling init fail");
        return ge::GRAPH_FAILED;
    }
    return tilingObj.SetKernelTiling();
}

static ge::graphStatus TilingPrepare4MoeFusedAddTopk(gert::TilingParseContext* context) {
    auto compileInfo = context->GetCompiledInfo<MoeFusedAddTopkCompileInfo>();
    // OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize;

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->ubSize = ubSize;

    // OP_TILING_CHECK(compileInfo->coreNum <= 0,
    //                 VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
    //                                                 "MoeFusedAddTopk GetHardwareInfo Failed, vectorCoreNum: %u",
    //                                                 compileInfo->coreNum),
    //                 return ge::GRAPH_FAILED);
    // OP_TILING_CHECK(compileInfo->ubSize <= 0,
    //                 VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
    //                                                 "MoeFusedAddTopk GetHardwareInfo Failed, ubSize: %lu",
    //                                                 compileInfo->ubSize),
    //                 return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class MoeFusedAddTopk : public OpDef {
public:
    explicit MoeFusedAddTopk(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("add_num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mapping_num")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mapping_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("group_num").AttrType(REQUIRED).Int(1);
        this->Attr("group_topk").AttrType(REQUIRED).Int(1);
        this->Attr("top_n").AttrType(REQUIRED).Int(1);
        this->Attr("top_k").AttrType(REQUIRED).Int(1);
        this->Attr("activate_type").AttrType(REQUIRED).Int(1);
        this->Attr("is_norm").AttrType(REQUIRED).Bool(true);
        this->Attr("scale").AttrType(REQUIRED).Float(1.0);
        this->Attr("enable_expert_mapping").AttrType(REQUIRED).Bool(false);

        this->AICore().SetTiling(optiling::TilingMoeFusedAddTopk);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(MoeFusedAddTopk);
}  // namespace ops
