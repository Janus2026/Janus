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
#include "kernel_operator_list_tensor_intf.h"

namespace kernels {

constexpr uint32_t TOTAL_UB_SIZE = 190 * 1024;
constexpr uint32_t KEY_UB_OFFSET = 0;
constexpr uint32_t KEY_UB_LEN = 95 * 1024;
constexpr uint32_t VALUE_UB_OFFSET = 95 * 1024;
constexpr uint32_t VALUE_UB_LEN = 95 * 1024;
constexpr int8_t TASK_DIRECTION_UP = 1;   // Forward task, src_beam_index > dst_beam_index
constexpr int8_t TASK_DIRECTION_DOWN = -1; // Backward task, src_beam_index < dst_beam_index

using namespace AscendC;

struct beamParams {
    int32_t b_idx{0};   // batch index
    int32_t beam_idx{0};  // beam index
    int32_t beam_token_num{0};  // number of tokens in beam
    int32_t beam_inner_offset;  // start offset of current beam writeout
    int8_t task_type;
};

template <class T1, class T2>
class SelectUnsharedKVKernel {
public:
    SelectUnsharedKVTilingData tiling_;
    AscendC::TPipe *pipe_ = nullptr;

    GlobalTensor<T2> beam_index_gm_;
    GlobalTensor<T2> group_token_num_gm_;
    GlobalTensor<T2> block_table_gm_;
    ListTensorDesc x_key_block_list;
    ListTensorDesc x_value_block_list;

    TBuf<TPosition::VECCALC> ubuf_;

    uint32_t core_id;
    uint32_t used_core_num;
    int32_t global_batch_idx = 0;
    int32_t global_beam_idx;
    int32_t batch;
    T2 beam_size_val;
    uint64_t block_batch_stride;
    T2 decode_step_val;

    __aicore__ inline SelectUnsharedKVKernel(AscendC::TPipe *pipe) {pipe_ = pipe;}

    __aicore__ inline void Init(GM_ADDR beam_index, GM_ADDR x_key_block, GM_ADDR x_value_block,
                                GM_ADDR block_table, GM_ADDR group_token_num,
                                GM_ADDR workspace, const SelectUnsharedKVTilingData *tiling)
    {
        tiling_ = *tiling;
        core_id = GetBlockIdx();
        used_core_num = GetBlockNum();
        beam_index_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T2 *>(beam_index));
        group_token_num_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T2 *>(group_token_num));
        x_key_block_list.Init((__gm__ void*)x_key_block);
        x_value_block_list.Init((__gm__ void*)x_value_block);
        block_table_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T2 *>(block_table));
        

        pipe_->InitBuffer(ubuf_, TOTAL_UB_SIZE);

        decode_step_val = tiling_.decode_step;
        beam_size_val = tiling_.beam_size;
        batch = tiling_.batch;
        block_batch_stride = tiling_.block_batch_stride;
    }

    __aicore__ inline bool calcValidBlockNew(beamParams &beam_params, int8_t task_type)
    {
        int64_t calc_block = 0;
        int32_t start_core_id = 0;
        while (calc_block < used_core_num) {
            if (global_batch_idx >= batch) {
                break;
            }

            int32_t left_group_num = task_type == TASK_DIRECTION_UP ? (beam_size_val - global_beam_idx) : (global_beam_idx + 1);
            int32_t valid_num = left_group_num > used_core_num ? used_core_num : left_group_num;

            if (core_id >= start_core_id && core_id - start_core_id < valid_num) {
                int32_t offset = core_id - start_core_id;
                int32_t beam_idx = task_type == TASK_DIRECTION_UP ? (global_beam_idx + offset) : (global_beam_idx - offset);
                uint32_t beam_gm_offset = global_batch_idx * beam_size_val + beam_idx;
                int32_t beam_token_num = beam_idx == 0 ? group_token_num_gm_.GetValue(beam_gm_offset)
                                         : group_token_num_gm_.GetValue(beam_gm_offset) - group_token_num_gm_(beam_gm_offset - 1);
                int32_t beam_inner_offset = beam_idx == 0 ? 0 : group_token_num_gm_.GetValue(beam_gm_offset - 1);
                bool has_task = true;

                if ((task_type == TASK_DIRECTION_UP && (beam_idx <= beam_inner_offset)) ||
                    (task_type == TASK_DIRECTION_DOWN && (beam_idx > (beam_inner_offset + beam_token_num - 1)))) {
                    has_task = false;
                }

                if (beam_token_num != 0 && has_task) {
                    beam_params.b_idx = global_batch_idx;
                    beam_params.beam_idx = beam_idx;
                    // Calculate the number of tasks actually executed and the start offset of the write address based on the task type
                    if (task_type == TASK_DIRECTION_UP) {
                        beam_params.beam_inner_offset = beam_inner_offset;
                        beam_params.beam_token_num = (beam_inner_offset + beam_token_num) > beam_params.beam_idx ?
                                                    (beam_params.beam_idx - 1 - beam_inner_offset + 1) : beam_token_num;
                    } else {
                        beam_params.beam_inner_offset = beam_inner_offset > beam_params.beam_idx ? beam_inner_offset : (beam_params.beam_idx + 1);
                        beam_params.beam_token_num = (beam_inner_offset + beam_token_num - beam_params.beam_inner_offset);
                    }
                    beam_params.task_type = task_type;
                }
            }

            int32_t process_group_num = used_core_num - start_core_id;
            calc_block += process_group_num;
            start_core_id += process_group_num;

            if (task_type == TASK_DIRECTION_UP) {
                global_beam_idx += process_group_num;
                if (global_beam_idx > beam_size_val - 1) {
                    global_batch_idx++;
                    global_beam_idx = 0;
                }
            } else {
                global_beam_idx -= process_group_num;
                if (global_beam_idx < 0) {
                    global_batch_idx++;
                    global_beam_idx = beam_size_val - 1;
                }
            }
        }
        return calc_block > 0 ? true : false;
    }

    __aicore__ inline bool calcValidBlock(beamParams &beam_params, int8_t task_type)
    {
        int64_t calc_block = 0;
        int32_t max_beam_token_num = 0;
        while (calc_block < used_core_num) {
            if (global_batch_idx >= batch) {
                break;
            }
            uint32_t beam_gm_offset = global_batch_idx * beam_size_val + global_beam_idx;
            // Get current beam index and beam number
            int32_t beam_token_num = global_beam_idx == 0 ? group_token_num_gm_.GetValue(beam_gm_offset)
                                    : group_token_num_gm_.GetValue(beam_gm_offset) - group_token_num_gm_.GetValue(beam_gm_offset - 1);
            int32_t beam_inner_offset = global_beam_idx == 0 ? 0 : group_token_num_gm_.GetValue(beam_gm_offset - 1);
            
            // Check if there is a task
            bool has_task = true;
            // Check if there is no task of the corresponding type
            if ((task_type == TASK_DIRECTION_UP && (global_beam_idx <= beam_inner_offset)) ||
                (task_type == TASK_DIRECTION_DOWN && (global_beam_idx > (beam_inner_offset + beam_token_num - 1)))) {
                has_task = false;
            }

            if (beam_token_num != 0 && has_task) {
                if (calc_block == core_id) {
                    beam_params.b_idx = global_batch_idx;
                    beam_params.beam_idx = global_beam_idx;
                    // Calculate the number of tasks actually executed and the start offset of the write address based on the task type
                    if (task_type == TASK_DIRECTION_UP) {
                        beam_params.beam_inner_offset = beam_inner_offset;
                        beam_params.beam_token_num = (beam_inner_offset + beam_token_num) > beam_params.beam_idx ?
                                                    (beam_params.beam_idx - 1 - beam_inner_offset + 1) : beam_token_num;
                    } else {
                        beam_params.beam_inner_offset = beam_inner_offset > beam_params.beam_idx ? beam_inner_offset : (beam_params.beam_idx + 1);
                        beam_params.beam_token_num = (beam_inner_offset + beam_token_num - beam_params.beam_inner_offset);
                    }
                    beam_params.task_type = task_type;
                }
                calc_block++;
            }
            
            if (task_type == TASK_DIRECTION_UP) {
                global_beam_idx++;
                if (global_beam_idx > beam_size_val - 1) {
                    global_batch_idx++;
                    global_beam_idx = 0;
                }
            } else {
                global_beam_idx--;
                if (global_beam_idx < 0) {
                    global_batch_idx++;
                    global_beam_idx = beam_size_val - 1;
                }
            }
        }
        return calc_block > 0 ? true : false;
    }

    __aicore__ inline void process()
    {
        AscendC::TEventID event_id_mte2_to_mte3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3);
        AscendC::TEventID event_id_mte3_to_mte2 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2);
        // Forward processing + backward processing
        int8_t loop_num = 2;

        for (size_t loop_idx = 0; loop_idx < loop_num; loop_idx++) {
            int8_t task_type = loop_idx == 0 ? TASK_DIRECTION_UP : TASK_DIRECTION_DOWN;

            global_batch_idx = 0;
            global_beam_idx = loop_idx == 0 ? 0 : (beam_size_val - 1);

            while (true) {
                beamParams beam_params;
                bool has_block = calcValidBlockNew(beam_params, task_type);

                if (!has_block) {
                    break;
                }

                int32_t src_beam_idx = beam_params.beam_idx;
                int32_t block_idx = block_table_gm_.GetValue(beam_params.b_idx);
                uint64_t beam_src_offset = block_idx * block_batch_stride +
                                    src_beam_idx * tiling_.block_beam_stride;
                LocalTensor<T1> key_ub_ = ubuf_.GetWithOffset<T1>(KEY_UB_LEN / sizeof(T1), KEY_UB_OFFSET);
                LocalTensor<T1> value_ub_ = ubuf_.GetWithOffset<T1>(VALUE_UB_LEN / sizeof(T1), VALUE_UB_OFFSET);
                // head_num split
                for (size_t i = 0; i < tiling_.copy_repeat_times; i++) {
                    uint32_t copy_head_num = (i == tiling_.copy_repeat_times - 1) ? tiling_.copy_head_num_tail : tiling_.copy_head_num_per_loop;
                    // BNSD input, use continuous multiple moves instead of non-continuous jump moves
                    uint32_t copy_len = copy_head_num * tiling_.max_decode_step * tiling_.head_dim;
                    uint32_t inner_offset = i * tiling_.copy_head_num_per_loop * tiling_.max_decode_step * tiling_.head_dim;
                    uint64_t head_src_offset = beam_src_offset + inner_offset;

                    for (size_t layer_idx = 0; layer_idx < tiling_.layer_num; layer_idx++) {
                        
                        GM_ADDR x_key_block_addr = (__gm__ uint8_t*)x_key_block_list.GetDataPtr<__gm__ uint8_t>(layer_idx);
                        GM_ADDR x_value_block_addr = (__gm__ uint8_t*)x_value_block_list.GetDataPtr<__gm__ uint8_t>(layer_idx);
                        GlobalTensor<T1> x_key_block_gm_;
                        x_key_block_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T1 *>(x_key_block_addr));
                        GlobalTensor<T1> x_value_block_gm_;
                        x_value_block_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T1 *>(x_value_block_addr));

                        if (beam_params.beam_token_num > 0) {
                            SetFlag<HardEvent::MTE3_MTE2>(event_id_mte3_to_mte2);
                            WaitFlag<HardEvent::MTE3_MTE2>(event_id_mte3_to_mte2);
                            DataCopy(key_ub_, x_key_block_gm_[head_src_offset], copy_len);
                            DataCopy(value_ub_, x_value_block_gm_[head_src_offset], copy_len);
                        }
                        // Inter-core read synchronization to prevent overwrite when writing
                        CrossCoreSetFlag<0x0, PIPE_MTE2>(0x8);
                        CrossCoreWaitFlag(0x8);
                        if (beam_params.beam_token_num > 0) {
                            SetFlag<HardEvent::MTE2_MTE3>(event_id_mte2_to_mte3);
                            WaitFlag<HardEvent::MTE2_MTE3>(event_id_mte2_to_mte3);
                            for (size_t i = 0; i < beam_params.beam_token_num; i++) {
                                // Copy out
                                int32_t dst_beam_idx = beam_params.beam_inner_offset + i;
                                uint64_t dst_offset = block_idx * block_batch_stride +
                                                    dst_beam_idx * tiling_.block_beam_stride + 
                                                    inner_offset;

                                if (src_beam_idx != dst_beam_idx) {
                                    DataCopy(x_key_block_gm_[dst_offset], key_ub_, copy_len);
                                    DataCopy(x_value_block_gm_[dst_offset], value_ub_, copy_len);
                                }
                            }
                        }
                    }
                }
            }
        }

    }
};
}
