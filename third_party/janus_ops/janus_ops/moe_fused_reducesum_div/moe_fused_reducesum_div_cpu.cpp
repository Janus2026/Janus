/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_fused_reducesum_div_cpu.cpp
 * \brief
 */
#include "moe_fused_reducesum_div_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr uint64_t UB_RESERVED_LENGTH = 2048;
constexpr uint64_t BYTE_ALIGN = 32;

int32_t align2(int32_t rows) {
    return (rows & 1) ? rows - 1 : rows;
}

int32_t align32(int32_t rows, int32_t elemBytes) {
    int32_t minElem = 32 / elemBytes;
    int32_t aligned = (rows + minElem - 1) / minElem * minElem;
    return aligned;
}


int32_t GetBytesFromDType(const ge::DataType& dtype) {
    switch (dtype) {
        case ge::DataType::DT_FLOAT:
            return sizeof(int32_t);
        case ge::DataType::DT_BF16:
            return sizeof(int16_t);
        case ge::DataType::DT_FLOAT16:
            return sizeof(int16_t);
        default:
            return 0;
    }
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    MoeFusedReducesumDivTilingData tiling;

    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    auto inputType = context->GetInputTensor(0)->GetDataType();
    auto dataBytes = GetBytesFromDType(inputType);
    if (dataBytes == 0) {
        std::cout << "[OPSERROR][reducesum_realdiv]The input type is not supported"  << std::endl;
        return ge::GRAPH_FAILED;
    }
    
    int32_t m = 0;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum()-1; i++){
        m += x1_shape->GetStorageShape().GetDim(i);
    }
    int32_t n = x1_shape->GetStorageShape().GetDim(x1_shape->GetStorageShape().GetDimNum() - 1);

    int32_t nAlign = align32(n, dataBytes);

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize -= UB_RESERVED_LENGTH;

    uint32_t coreNum = platformInfo.GetCoreNumAiv();

    int32_t singleCoreCalcRows = ubSize / (sizeof(float) * (2 * nAlign + 1)); // input output sum
    singleCoreCalcRows = align2(singleCoreCalcRows);

    int32_t useCoreNum = m > coreNum ? coreNum : m;

    int32_t bigCoreNum = m % useCoreNum;
    int32_t littleCoreNum = useCoreNum - bigCoreNum;
    int32_t avgCoreCalcRows = m / useCoreNum;

    context->SetBlockDim(useCoreNum);

    tiling.set_n(n);
    tiling.set_nAlign(nAlign);
    tiling.set_bigCoreNum(bigCoreNum);
    tiling.set_littleCoreNum(littleCoreNum);
    tiling.set_avgCoreCalcRows(avgCoreCalcRows);
    tiling.set_singleCoreCalcRows(singleCoreCalcRows);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

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
class MoeFusedReducesumDiv : public OpDef {
public:
    explicit MoeFusedReducesumDiv(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

    }
};

OP_ADD(MoeFusedReducesumDiv);
}
