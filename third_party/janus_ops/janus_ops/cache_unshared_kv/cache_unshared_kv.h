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

#pragma once

#include "kernel_operator.h"

namespace kernels {

constexpr uint32_t TOTAL_UB_SIZE = 190 * 1024;
constexpr uint32_t KEY_UB_OFFSET = 0;
constexpr uint32_t KEY_UB_LEN = 95 * 1024;
constexpr uint32_t VALUE_UB_OFFSET = 95 * 1024;
constexpr uint32_t VALUE_UB_LEN = 95 * 1024;

using namespace AscendC;

template <class T>
class CacheUnsharedKvKernel {
public:
    CacheUnsharedKvTilingData tiling_;
    AscendC::TPipe *pipe_ = nullptr;

    GlobalTensor<T> x_key_block_gm_;
    GlobalTensor<T> x_value_block_gm_;
    GlobalTensor<T> cur_key_gm_;
    GlobalTensor<T> cur_value_gm_;
    GlobalTensor<int32_t> block_table_gm_;
    GlobalTensor<int32_t> decode_step_gm_;
    GlobalTensor<T> select_key_block_gm_;
    GlobalTensor<T> select_value_block_gm_;

    TBuf<TPosition::VECCALC> ubuf_;

    uint32_t core_id;
    uint32_t used_core_num;
    int32_t decode_step_val;

    __aicore__ inline CacheUnsharedKvKernel(AscendC::TPipe *pipe) {pipe_ = pipe;}

    __aicore__ inline void Init(GM_ADDR x_key_block, GM_ADDR x_value_block,
                                GM_ADDR cur_key, GM_ADDR cur_value, GM_ADDR block_table, GM_ADDR decode_step,
                                const CacheUnsharedKvTilingData *tiling,
                                GM_ADDR select_key_block, GM_ADDR select_value_block)
    {
        tiling_ = *tiling;
        core_id = GetBlockIdx();
        used_core_num = GetBlockNum();
        x_key_block_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x_key_block));
        x_value_block_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x_value_block));
        cur_key_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(cur_key));
        cur_value_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(cur_value));
        block_table_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_table));
        select_key_block_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(select_key_block));
        select_value_block_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(select_value_block));
        pipe_->InitBuffer(ubuf_, TOTAL_UB_SIZE);
        decode_step_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(decode_step));
        decode_step_val = decode_step_gm_.GetValue(0);
    }

    __aicore__ inline void process()
    {
        AscendC::TEventID event_id_mte2_to_mte3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3);
        AscendC::TEventID event_id_mte3_to_mte2 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2);

        LocalTensor<T> key_ub_ = ubuf_.GetWithOffset<T>(KEY_UB_LEN / sizeof(T), KEY_UB_OFFSET);
        LocalTensor<T> value_ub_ = ubuf_.GetWithOffset<T>(VALUE_UB_LEN / sizeof(T), VALUE_UB_OFFSET);
        // Task layer loop
        for (size_t task_idx = core_id; task_idx < tiling_.total_task; task_idx+=used_core_num)
        {
            uint32_t batch_idx = task_idx / tiling_.task_num_per_batch;
            uint32_t inner_idx = task_idx % tiling_.task_num_per_batch;
            uint32_t block_idx = block_table_gm_.GetValue(batch_idx);
            uint32_t copy_tokens = (inner_idx == tiling_.task_num_per_batch - 1) ? 
                                   tiling_.copy_beam_tail : 
                                   tiling_.copy_beam_per_task;
            uint32_t beam_src_offset = batch_idx * tiling_.beam_size * tiling_.head_num * tiling_.head_dim +
                                       inner_idx * tiling_.copy_beam_per_task * tiling_.head_num * tiling_.head_dim;
            uint32_t beam_dst_offset = block_idx * tiling_.block_batch_stride + inner_idx * tiling_.copy_beam_per_task * tiling_.block_beam_stride;

            // head_num loop
            for (size_t inner_loop_idx = 0; inner_loop_idx < tiling_.copy_repeat_times; inner_loop_idx++)
            {
                uint32_t copy_head_num = (inner_loop_idx == tiling_.copy_repeat_times - 1) ? 
                                        tiling_.copy_head_num_tail : 
                                        tiling_.copy_head_num_per_loop;
                uint32_t inner_src_offset = inner_loop_idx * tiling_.copy_head_num_per_loop * tiling_.head_dim;
                uint32_t src_offset = beam_src_offset + inner_src_offset;
                uint32_t copy_len = copy_tokens * copy_head_num * tiling_.head_dim;
                // Copy in
                SetFlag<HardEvent::MTE3_MTE2>(event_id_mte3_to_mte2);
                WaitFlag<HardEvent::MTE3_MTE2>(event_id_mte3_to_mte2);
                DataCopy(key_ub_, cur_key_gm_[src_offset], copy_len);
                DataCopy(value_ub_, cur_value_gm_[src_offset], copy_len);

                SetFlag<HardEvent::MTE2_MTE3>(event_id_mte2_to_mte3);
                WaitFlag<HardEvent::MTE2_MTE3>(event_id_mte2_to_mte3);

                uint32_t dst_decode_offset = (decode_step_val - 1) * tiling_.head_dim;
                uint32_t dst_head_offset = inner_loop_idx * tiling_.copy_head_num_per_loop * tiling_.block_head_stride;
                uint32_t dst_offset = beam_dst_offset + dst_decode_offset + dst_head_offset;
                DataCopyParams copy_out_params;
                copy_out_params.blockCount = static_cast<uint16_t>(copy_head_num * copy_tokens);
                copy_out_params.blockLen = static_cast<uint16_t>(tiling_.head_dim * sizeof(T) / 32);
                copy_out_params.srcStride = 0;
                copy_out_params.dstStride = static_cast<uint16_t>((tiling_.block_head_stride - tiling_.head_dim) * sizeof(T) / 32);
                DataCopy(select_key_block_gm_[dst_offset], key_ub_, copy_out_params);
                DataCopy(select_value_block_gm_[dst_offset], value_ub_, copy_out_params);
            }
        }
    }
};
}
