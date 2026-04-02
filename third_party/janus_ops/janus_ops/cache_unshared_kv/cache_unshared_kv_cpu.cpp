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

#include "cache_unshared_kv_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


enum InputIndex {
    X_KEY_BLOCK = 0,
    X_VALUE_BLOCK,
    CURR_KEY,
    CURR_VALUE,
    BLOCK_TABLE,
    DECODE_STEP
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
    
    CacheUnsharedKvTilingData tiling;
    const gert::StorageShape* x_key_block_shape = context->GetInputShape(X_KEY_BLOCK);
    const gert::StorageShape* block_table_shape = context->GetInputShape(BLOCK_TABLE);
    uint32_t block_num = x_key_block_shape->GetStorageShape().GetDim(0);
    uint32_t beam_size = x_key_block_shape->GetStorageShape().GetDim(1);
    uint32_t max_decode_step = x_key_block_shape->GetStorageShape().GetDim(3);
    uint32_t head_num = x_key_block_shape->GetStorageShape().GetDim(2);
    uint32_t head_dim = x_key_block_shape->GetStorageShape().GetDim(4);
    uint32_t batch = block_table_shape->GetStorageShape().GetDim(0);

    // Inter-core partitioning along the beam direction, intra-core partitioning along the N direction
    // Calculate the maximum single task based on UB utilization maximization
    uint32_t copy_head_num_per_loop = MAX_USED_UB_SIZE / sizeof(int16_t) / head_dim;
    uint32_t copy_repeat_times = (head_num + copy_head_num_per_loop - 1) / copy_head_num_per_loop;
    uint32_t copy_head_num_tail = head_num % copy_head_num_per_loop;
    uint32_t copy_beam_per_task = 1;

    if (copy_head_num_per_loop > head_num) {
        copy_head_num_per_loop = head_num;
        copy_beam_per_task = MAX_USED_UB_SIZE / (head_num * head_dim * sizeof(int16_t));
    }

    if (copy_head_num_tail == 0) {
        copy_head_num_tail = copy_head_num_per_loop;
    }

    uint32_t copy_beam_tail = beam_size % copy_beam_per_task;

    if (copy_beam_tail == 0) {
        copy_beam_tail = copy_beam_per_task;
    }

    uint32_t task_num_per_batch = (beam_size + copy_beam_per_task - 1) / copy_beam_per_task;
    uint32_t total_task = task_num_per_batch * batch;
    uint32_t used_core_num = std::min(total_task, core_num);
    uint32_t block_head_stride = max_decode_step * head_dim;
    uint32_t block_beam_stride = head_num * block_head_stride;
    uint32_t block_batch_stride = block_beam_stride * beam_size;

    // std::cout << "tiling:total_tokens " << total_tokens << std::endl;
    // std::cout << "tiling:batch " << batch << std::endl;
    // std::cout << "tiling:beam_size " << beam_size << std::endl;
    // std::cout << "tiling:head_num " << head_num << std::endl;
    // std::cout << "tiling:head_dim " << head_dim << std::endl;
    // std::cout << "tiling:decode_step " << decode_step << std::endl;
    // std::cout << "tiling:max_decode_step " << max_decode_step << std::endl;
    // std::cout << "tiling:total_task " << total_task << std::endl;
    // std::cout << "tiling:used_core_num " << used_core_num << std::endl;
    // std::cout << "tiling:copy_beam_per_task " << copy_beam_per_task << std::endl;
    // std::cout << "tiling:copy_head_num_per_loop " << copy_head_num_per_loop << std::endl;
    // std::cout << "tiling:copy_repeat_times " << copy_repeat_times << std::endl;
    // std::cout << "tiling:copy_head_num_tail " << copy_head_num_tail << std::endl;

    tiling.set_batch(batch);
    tiling.set_beam_size(beam_size);
    tiling.set_head_num(head_num);
    tiling.set_head_dim(head_dim);
    tiling.set_max_decode_step(max_decode_step);
    tiling.set_used_core_num(used_core_num);
    tiling.set_block_beam_stride(block_beam_stride);
    tiling.set_block_head_stride(block_head_stride);
    tiling.set_copy_head_num_per_loop(copy_head_num_per_loop);
    tiling.set_copy_repeat_times(copy_repeat_times);
    tiling.set_copy_head_num_tail(copy_head_num_tail);
    tiling.set_copy_beam_per_task(copy_beam_per_task);
    tiling.set_total_task(total_task);
    tiling.set_task_num_per_batch(task_num_per_batch);
    tiling.set_block_batch_stride(block_batch_stride);
    tiling.set_copy_beam_tail(copy_beam_tail);

    context->SetBlockDim(used_core_num);
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
class CacheUnsharedKv : public OpDef {
public:
    explicit CacheUnsharedKv(const char* name) : OpDef(name)
    {
        this->Input("x_key_block")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x_value_block")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("curr_key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("curr_value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});    
        this->Input("decode_step")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("select_key_block")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("select_value_block")
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

OP_ADD(CacheUnsharedKv);
}
