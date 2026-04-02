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

#include "select_unshared_kv_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


enum InputIndex {
    BEAM_INDEX = 0,
    BLOCK_TABLE,
    X_KEY_BLOCK,
    X_VALUE_BLOCK,
    GROUP_TOKEN_NUM
};

constexpr uint64_t DIM_0 = 0;
constexpr uint64_t DIM_1 = 1;
constexpr uint64_t DIM_2 = 2;
constexpr uint64_t DIM_3 = 3;
// ub is 190K, kv is halved
constexpr uint64_t MAX_USED_UB_SIZE = 95 * 1024;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // Get hardware information
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t core_num = ascendcPlatform.GetCoreNumAiv();

    SelectUnsharedKVTilingData tiling;
    const gert::StorageShape* x_key_block_shape = context->GetDynamicInputShape(X_KEY_BLOCK, 0);
    const gert::StorageShape* block_table_shape = context->GetInputShape(BLOCK_TABLE);

    uint32_t block_num = x_key_block_shape->GetStorageShape().GetDim(0);
    uint32_t head_num = x_key_block_shape->GetStorageShape().GetDim(2);
    uint32_t max_decode_step = x_key_block_shape->GetStorageShape().GetDim(3);
    uint32_t head_dim = x_key_block_shape->GetStorageShape().GetDim(4);
    uint32_t decode_step = static_cast<uint32_t>(*(context->GetAttrs()->GetAttrPointer<int>(0)));
    uint32_t beam_size = static_cast<uint32_t>(*(context->GetAttrs()->GetAttrPointer<int>(1)));
    uint32_t layer_num = static_cast<uint32_t>(*(context->GetAttrs()->GetAttrPointer<int>(2)));

    uint32_t batch = block_table_shape->GetStorageShape().GetDim(0);
    uint32_t total_beam = batch * beam_size;
    uint32_t used_core_num = std::min(total_beam, core_num);
    uint64_t block_head_stride = max_decode_step * head_dim;
    uint64_t block_beam_stride = head_num * block_head_stride;
    uint64_t block_batch_stride = block_beam_stride * beam_size;
    uint32_t copy_head_num_per_loop = MAX_USED_UB_SIZE / sizeof(int16_t) / head_dim;
    uint32_t copy_repeat_times = (head_num + copy_head_num_per_loop - 1) / copy_head_num_per_loop;
    uint32_t copy_head_num_tail = head_num % copy_head_num_per_loop;
    
    if (head_num < copy_head_num_per_loop) {
        copy_head_num_per_loop = head_num;
    }

    if (copy_head_num_tail == 0) {
        copy_head_num_tail = copy_head_num_per_loop;
    }

    tiling.set_total_beam(total_beam);
    tiling.set_head_num(head_num);
    tiling.set_head_dim(head_dim);
    tiling.set_max_decode_step(max_decode_step);
    tiling.set_used_core_num(used_core_num);
    tiling.set_block_beam_stride(block_beam_stride);
    tiling.set_block_batch_stride(block_batch_stride);
    tiling.set_block_head_stride(block_head_stride);
    tiling.set_copy_head_num_per_loop(copy_head_num_per_loop);
    tiling.set_copy_repeat_times(copy_repeat_times);
    tiling.set_copy_head_num_tail(copy_head_num_tail);
    tiling.set_decode_step(decode_step);
    tiling.set_beam_size(beam_size);
    tiling.set_batch(batch);
    tiling.set_layer_num(layer_num);
    
    context->SetBlockDim(used_core_num);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    uint32_t layer_num = static_cast<uint32_t>(*(context->GetAttrs()->GetAttrPointer<int>(2)));
    for (size_t i = 0; i < layer_num; i++) {
        const gert::Shape* key_block_shape = context->GetDynamicInputShape(X_KEY_BLOCK, i);
        const gert::Shape* value_block_shape = context->GetDynamicInputShape(X_VALUE_BLOCK, i);
        gert::Shape* select_key_shape = context->GetOutputShape(i);
        gert::Shape* select_value_shape = context->GetOutputShape(i + layer_num);
        *select_key_shape = *key_block_shape;
        *select_value_shape = *value_block_shape;
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    uint32_t layer_num = static_cast<uint32_t>(*(context->GetAttrs()->GetAttrPointer<int>(2)));
    const auto inputDataType = context->GetDynamicInputDataType(X_KEY_BLOCK, 0);
    for (size_t i = 0; i < layer_num; i++) {
        context->SetOutputDataType(i, inputDataType);
        context->SetOutputDataType(i + layer_num, inputDataType);
    }
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class SelectUnsharedKV : public OpDef {
public:
    explicit SelectUnsharedKV(const char* name) : OpDef(name)
    {
        this->Input("beam_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x_key_block")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x_value_block")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("group_token_num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("select_key_block")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("select_value_block")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("decode_step").Int(0);
        this->Attr("beam_size").Int(0);
        this->Attr("layer_num").Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

    }
};

OP_ADD(SelectUnsharedKV);
}
