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

#include "kernel_operator.h"

__aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t y = 16)
{
    return (x + y - 1) / y * y;
}

constexpr int32_t BLOCK_SIZE = 32;
template <typename T> class ConvertKvCacheFormat {
public:
    __aicore__ inline ConvertKvCacheFormat() {}
    __aicore__ __attribute__((always_inline)) inline void
    SetArgs(__gm__ uint8_t *__restrict__ k_cache_ptr_in_gm, __gm__ uint8_t *__restrict__ v_cache_ptr_in_gm,
            __gm__ uint8_t *__restrict__ kv_cache_offset_in_gm, __gm__ uint8_t *__restrict__ kv_seq_len_in_gm,
            __gm__ uint8_t *__restrict__ tiling_para_gm)
    {
        k_cache_ptr_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k_cache_ptr_in_gm));
        v_cache_ptr_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(v_cache_ptr_in_gm));
        kv_cache_offset_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(kv_cache_offset_in_gm));
        kv_seq_len_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(kv_seq_len_in_gm));
        is_prefill = (bool)(*((__gm__ uint32_t *)tiling_para_gm));
        num_batches = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + 1));
        num_kv_heads = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + 2));
        head_size_k = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + 3));
        head_size_v = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + 4));
        block_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + 5));
        token_size_k = num_kv_heads * head_size_k;
        token_size_v = num_kv_heads * head_size_v;
        uint32_t token_size = token_size_k > token_size_v ? token_size_k : token_size_v;
        pipe.InitBuffer(ub_buffer, RoundUp(block_size * token_size * sizeof(T), BLOCK_SIZE));

        ub_local = ub_buffer.Get<T>();
        nd2NzParamsInK = {1, block_size, token_size_k, 0, token_size_k, block_size, 1, 0};
        nd2NzParamsInV = {1, block_size, token_size_v, 0, token_size_v, block_size, 1, 0};
        copyParamsOutK = {1, static_cast<uint16_t>(block_size * token_size_k * sizeof(T) / BLOCK_SIZE), 0, 0};
        copyParamsOutV = {1, static_cast<uint16_t>(block_size * token_size_v * sizeof(T) / BLOCK_SIZE), 0, 0};
    }
    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        uint64_t cur_batch = 0;
        uint32_t process_num = num_batches;
        for (uint32_t process = block_idx; process < process_num; process += (uint32_t)block_num) {
            cur_batch = process;
            if (cur_batch >= num_batches)
                break;
            uint32_t kv_seqlen = kv_seq_len_gm.GetValue(cur_batch);
            if (kv_seqlen == 0) {
                continue;
            }
            uint32_t cur_kv_seqlen = kv_seqlen;
            if (is_prefill) {
                ConvertKvCacheND2NZForPrefill(cur_batch, cur_kv_seqlen);
            } else {
                ConvertKvCacheND2NZForDecode(cur_batch, cur_kv_seqlen);
            }
        }
    }
    // Perform nd2nz conversion every 128 tokens; in prefill stage it needs to be done many times,
    // while in decode stage it only needs to be done once.
    // Need to check whether the tail block length exceeds 128.
    // During prefill, check from front to back whether blocks meet 128.
    __aicore__ __attribute__((always_inline)) inline void ConvertKvCacheND2NZForPrefill(uint32_t cur_batch,
                                                                                        uint32_t cur_kv_seqlen)
    {
        uint64_t cur_kv_cache_offset = kv_cache_offset_gm.GetValue(cur_batch);
        auto cur_k_cache_ptr = k_cache_ptr_gm[cur_kv_cache_offset * token_size_k];
        auto cur_v_cache_ptr = v_cache_ptr_gm[cur_kv_cache_offset * token_size_v];
        uint64_t cache_start_k = 0;
        uint64_t cache_start_v = 0;
        while (cur_kv_seqlen >= block_size) {
            auto new_k_cache_ptr = cur_k_cache_ptr[cache_start_k];
            auto new_v_cache_ptr = cur_v_cache_ptr[cache_start_v];
            ConvertKvCacheND2NZ(new_k_cache_ptr, ub_local, nd2NzParamsInK, copyParamsOutK);
            ConvertKvCacheND2NZ(new_v_cache_ptr, ub_local, nd2NzParamsInV, copyParamsOutV);
            cache_start_k += block_size * token_size_k;
            cache_start_v += block_size * token_size_v;
            cur_kv_seqlen -= block_size;
        }
    }
    // During decode, check from back to front whether blocks meet 128.
    __aicore__ __attribute__((always_inline)) inline void ConvertKvCacheND2NZForDecode(uint32_t cur_batch,
                                                                                       uint32_t cur_kv_seqlen)
    {
        uint64_t cur_kv_cache_offset = kv_cache_offset_gm.GetValue(cur_batch);
        auto cur_k_cache_ptr = k_cache_ptr_gm[cur_kv_cache_offset * token_size_k];
        auto cur_v_cache_ptr = v_cache_ptr_gm[cur_kv_cache_offset * token_size_v];
        if ((cur_kv_seqlen > 0) && (cur_kv_seqlen % block_size == 0)) {
            uint64_t cache_start_k = (cur_kv_seqlen - block_size) * token_size_k;
            uint64_t cache_start_v = (cur_kv_seqlen - block_size) * token_size_v;
            auto new_k_cache_ptr = cur_k_cache_ptr[cache_start_k];
            auto new_v_cache_ptr = cur_v_cache_ptr[cache_start_v];
            ConvertKvCacheND2NZ(new_k_cache_ptr, ub_local, nd2NzParamsInK, copyParamsOutK);
            ConvertKvCacheND2NZ(new_v_cache_ptr, ub_local, nd2NzParamsInV, copyParamsOutV);
        }
    }
    __aicore__ __attribute__((always_inline)) inline void ConvertKvCacheND2NZ(AscendC::GlobalTensor<T> &src,
                                                                                 AscendC::LocalTensor<T> &ubAddr,
                                                                                 AscendC::Nd2NzParams &nd2NzParamsIn,
                                                                                 AscendC::DataCopyParams &copyParamsOut)
    {
        // 1. gm to l1, nd -> nz
        DataCopy(ubAddr, src, nd2NzParamsIn);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
        // 2. l1 to gm, nz -> nz
        DataCopy(src, ubAddr, copyParamsOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
private:
    AscendC::GlobalTensor<T> k_cache_ptr_gm;
    AscendC::GlobalTensor<T> v_cache_ptr_gm;
    AscendC::GlobalTensor<int64_t> kv_cache_offset_gm;
    AscendC::GlobalTensor<int32_t> kv_seq_len_gm;
    bool is_prefill;
    uint32_t num_batches;
    uint32_t num_kv_heads;
    uint32_t head_size_k;
    uint32_t head_size_v;
    uint32_t block_size;
    uint32_t token_size_k;
    uint32_t token_size_v;
    AscendC::LocalTensor<T> ub_local;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub_buffer;
    AscendC::Nd2NzParams nd2NzParamsInK;
    AscendC::DataCopyParams copyParamsOutK;
    AscendC::Nd2NzParams nd2NzParamsInV;
    AscendC::DataCopyParams copyParamsOutV;
};

// k_cache_ptr: [max_tokens, num_kv_heads, head_size_k] (half)
// v_cache_ptr: [max_tokens, num_kv_heads, head_size_v] (half)
// kv_cache_offset: [num_batches] (int64) token offset for each batch
// kv_seq_len: [num_batches] (int32) valid token num for each batch
// tiling: [6] uint32_t
//   [0]: is_prefill (1: prefill, 0: decode)
//   [1]: num_batches
//   [2]: num_kv_heads
//   [3]: head_size_k
//   [4]: head_size_v
//   [5]: block_size (default 128)
extern "C" __global__ __aicore__ void
convert_kv_cache_format(GM_ADDR k_cache_ptr, GM_ADDR v_cache_ptr, GM_ADDR kv_cache_offset,
                        GM_ADDR kv_seq_len, GM_ADDR workspace, GM_ADDR tiling)
{
    if ASCEND_IS_AIV {
        ConvertKvCacheFormat<half> op;
        op.SetArgs(k_cache_ptr, v_cache_ptr, kv_cache_offset, kv_seq_len, tiling);
        op.Run();
    }
}