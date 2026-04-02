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

#include "mixkernels/include/common.h"
#include "mixkernels/include/common_func.h"
#include "mixkernels/include/simd.h"
#include "mixkernels/include/iterator.h"
#include "mixkernels/include/mma.h"
#include "mixkernels/include/utils.h"
#include "multi_latent_attention_npu.h"

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

// define common const value

// FFTS Flag
constexpr int32_t QK_READY = 0;
constexpr int32_t SOFTMAX_READY = 1;
constexpr int32_t UPDATE_READY = 2;
constexpr int32_t QK_READY_DECODER = 3;
constexpr int32_t SOFTMAX_READY_DECODER = 4;
constexpr int32_t UPDATE_READY_DECODER = 5;
constexpr int32_t TAIL_OPTIMIZATION_SYNC = 6;
constexpr int32_t TAIL_OPTIMIZATION_SYNC2 = 7;

constexpr int32_t BLOCK_SIZE_32 = 32;
constexpr int64_t TMP_SIZE = 65536;              // 256 * 256
constexpr int32_t BIT_SHIFT = 8;

const int32_t TILING_BATCH = 0;
const int32_t TILING_NUMHEADS = 1;
const int32_t TILING_HEADDIM = 2;
const int32_t TILING_NUMBLOKS = 3;
const int32_t TILING_BLOCKSIZE = 4;
const int32_t TILING_MAXBLOCKS = 5;
const int32_t TILING_TOR = 6;
const int32_t TILING_KVHEADS = 7;
const int32_t TILING_HEADSIZE = 8;
const int32_t TILING_PARASIZE = 9;

const int32_t TILING_MTP_HEAD_SPLIT_SIZE = 10;
const int32_t TILING_TOTAL_BLOCK_NUM = 11;
const int32_t TILING_MASK_TYPE_ND = 12;
const int32_t TILING_MAX_KV_SEQ_LEN = 14;

const int32_t TILING_BLOCKSIZE_CALC = 25;
const int32_t TILING_HEADDIM_K_SPLIT = 38;
const int32_t TILING_HEADDIM_V_SPLIT = 39;
const int32_t TILING_HEADDIM_V_SPLIT_VECTOR_FORMER = 40;
const int32_t TILING_HEADDIM_V_SPLIT_VECTOR_TAIL = 41;
constexpr uint32_t CONST_16 = 16;

const int32_t MASK_COLUMNS = 128;

template <typename T>
using GlobalT = AscendC::GlobalTensor<T>;

template <typename T>
using LocalT = AscendC::LocalTensor<T>;

enum class TilingKeyType {
    TILING_HALF_DATA = 0,
    TILING_BF16_DATA = 1,
    TILING_INT8_DATA = 2
};

enum class InputFormat {
    ND_FORMAT = 0,
    NZ_FORMAT = 1
};

enum class BlockStack {
    ONE_FLOW = 0,
    FOUR_FLOW = 1
};

template<TilingKeyType tilingKeyType>
struct AttentionType
{
};

template<>
struct AttentionType<TilingKeyType::TILING_HALF_DATA>
{
    using mm1OutputType = float;
    using mm1CopyType = float;
    using mmBiasType = float;
    using mmScaleType = float;
    using mm2OutputType = float;
    using mm2CopyType = float;
};

template<>
struct AttentionType<TilingKeyType::TILING_BF16_DATA>
{
    using mm1OutputType = float;
    using mm1CopyType = float;
    using mmBiasType = float;
    using mmScaleType = float;
    using mm2OutputType = float;
    using mm2CopyType = float;
};

template<>
struct AttentionType<TilingKeyType::TILING_INT8_DATA>
{
    using mm1OutputType = int32_t;
    using mm1CopyType = int32_t;
    using mmBiasType = float;
    using mmScaleType = float;
    using mm2OutputType = int32_t;
    using mm2CopyType = int32_t;
};

#ifdef __DAV_C220_CUBE__
constexpr int32_t L0AB_HALF_BUF_SIZE = 16384;    // 128 * 128 = 16K
constexpr int32_t L0AB_UINT8_BUF_SIZE = 16384 * 2;
constexpr int32_t L0C_FLOAT_BUF_SIZE = 16384;
constexpr int32_t L0C_UINT8_BUF_SIZE = 131072;
constexpr int32_t CUBE_MATRIX_SIZE = 256;        // 16 * 16
constexpr int64_t L0AB_UINT8_BLOCK_SIZE = 32768; // 128 * 128 * 2B
constexpr int32_t L1_HALF_BUF_SIZE = 65536;  // 256 * 256
constexpr int32_t L1_P_UINT8_BUF_SIZE = 32768;

constexpr int32_t TMP_SIZE_DECODER = 32768;
constexpr int32_t BLOCK_SIZE = 16;

constexpr int32_t L1_HALF_BUF_SIZE_DECODER = 16384;
constexpr int32_t L1_UINT8_BUF_SIZE_DECODER = 16384 * 2;
constexpr int32_t L1_KV_HALF_BUF_SIZE = 73728;// 2* 128 * 256
constexpr int32_t L1_KV_UINT8_BUF_SIZE = 73728 * 2;
constexpr uint64_t L1_E_UINT8_SIZE = 1024;  // 32 * 32 * 1B
constexpr uint64_t L1_SCALE_UINT8_SIZE = 4096;  // uint64 256 * 8 * 2head
constexpr uint64_t L1_SCALE_UINT64_SIZE = L1_SCALE_UINT8_SIZE / 8;
constexpr uint64_t L1_OFFSET_UINT8_SIZE = 2048;  // int32 256 * 4 8 2head
constexpr uint64_t L1_OFFSET_INT32_SIZE = L1_OFFSET_UINT8_SIZE / 4;

//DeQuant
constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN = 32768; // 32 KB
constexpr uint32_t L0C_PINGPONG_BUFFER_LEN_INT32 = 16384; // 65536 / 4
constexpr uint32_t CUBE_MATRIX_SIZE_512 = 16 * 32;       // 16 * 23
constexpr int32_t BLOCK_SIZE_16 = 16;
constexpr uint64_t CONST_4 = 4;
constexpr uint64_t CONST_32 = 32;
constexpr uint64_t CONST_64 = 64;
constexpr uint64_t CONST_128 = 128;
constexpr uint32_t EMBED_SPLIT = 256;
constexpr uint32_t ROUND_EMBED_SPLIT = 256;

#elif __DAV_C220_VEC__
constexpr uint32_t HALF_VECTOR_SIZE = 128;
constexpr uint32_t UB_ALIGN_BYTE = 32;
constexpr int64_t UB_UINT8_BLOCK_SIZE_MLA = 16384;      // 96 * 128 * 2B // prefill/decoder diff
constexpr int64_t UB_UINT8_BLOCK_SIZE_NORM = 24576;
constexpr int64_t UB_UINT8_LINE_SIZE = 512;         // 64 * 4B，double buffer
constexpr int64_t UB_HALF_LINE_SIZE = 256;          // UB_FLOAT_LINE_SIZE * 2
constexpr int64_t UB_FLOAT_LINE_SIZE = 64;         // 64，double buffer

constexpr int64_t PRE_UB_UINT8_BLOCK_SIZE = 16384;  // 64 * 128 * 2B
constexpr int32_t UB_HALF_BUF_SIZE = 8192;          // 64 * 128
constexpr int32_t TMP_SIZE_DECODER = 32768;
constexpr int32_t STAGE2_UB_UINT8_BLOCK_SIZE = 8192;
constexpr int32_t CUBE_MATRIX_SIZE = 256;
constexpr uint32_t MAX_UB_SIZE = 196608; // 192 * 1024
constexpr uint32_t EMBED_SPLIT_SM = 128;
constexpr uint32_t ROUND_EMBED_SPLIT_SM = 128;

__aicore__ __attribute__((always_inline)) void inline __set_mask(int32_t len)
{
    uint64_t mask = 0;
    uint64_t one = 1;
    uint64_t temp = len % FLOAT_VECTOR_SIZE;
    for (int64_t i = 0; i < temp; i++) {
        mask |= one << i;
    }

    if (len == VECTOR_SIZE) {
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    } else if (len >= FLOAT_VECTOR_SIZE) {
        SetVectorMask<int8_t>(mask, (uint64_t)-1);
    } else {
        SetVectorMask<int8_t>(0x0, mask);
    }
}

template<BlockStack blockStack>
struct UbufAlloc
{
};

template<>
struct UbufAlloc<BlockStack::ONE_FLOW>
{
    const uint32_t ls32_ubuf_offset = 0;
    const uint32_t ls32_quant_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t lp_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t lp32_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t mask_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t lo_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t mask32_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t ls16_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t lm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t hm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 1 * UB_UINT8_LINE_SIZE;
    const uint32_t pm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 2 * UB_UINT8_LINE_SIZE;
    const uint32_t pm32_ubuf_stage2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 3 * UB_UINT8_LINE_SIZE;
    const uint32_t descale1_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 4 * UB_UINT8_LINE_SIZE;
    const uint32_t descale2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 5 * UB_UINT8_LINE_SIZE;
    const uint32_t dm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 6 * UB_UINT8_LINE_SIZE;
    const uint32_t dm32_ubuf_stage2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 7 * UB_UINT8_LINE_SIZE;
    const uint32_t ll_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 9 * UB_UINT8_LINE_SIZE;
    const uint32_t ll_ubuf_stage2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 11 * UB_UINT8_LINE_SIZE;
    const uint32_t gm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 13 * UB_UINT8_LINE_SIZE;
    const uint32_t gl_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 15 * UB_UINT8_LINE_SIZE;
    const uint32_t gl32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 15 * UB_UINT8_LINE_SIZE;
    const uint32_t p_scale_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 17 * UB_UINT8_LINE_SIZE;
    const uint32_t go_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t go32_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t tv32_ubuf_offset = 10 * UB_UINT8_BLOCK_SIZE_MLA;
};

template<>
struct UbufAlloc<BlockStack::FOUR_FLOW>
{
    const uint32_t ls32_ubuf_offset = 0;
    const uint32_t ls32_quant_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t lp_ubuf_offset = 0 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t lp32_ubuf_offset = 0 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t mask_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t lo_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t mask32_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t ls16_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t lm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t hm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 1 * UB_UINT8_LINE_SIZE;
    const uint32_t pm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 2 * UB_UINT8_LINE_SIZE;
    const uint32_t pm32_ubuf_stage2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 3 * UB_UINT8_LINE_SIZE;
    const uint32_t descale1_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 4 * UB_UINT8_LINE_SIZE;
    const uint32_t descale2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 5 * UB_UINT8_LINE_SIZE;
    const uint32_t dm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 6 * UB_UINT8_LINE_SIZE;
    const uint32_t dm32_ubuf_stage2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 7 * UB_UINT8_LINE_SIZE;
    const uint32_t ll_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 10 * UB_UINT8_LINE_SIZE;
    const uint32_t ll_ubuf_stage2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 12 * UB_UINT8_LINE_SIZE;
    const uint32_t gm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 14 * UB_UINT8_LINE_SIZE;
    const uint32_t gl_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 16 * UB_UINT8_LINE_SIZE;
    const uint32_t gl32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA + 16 * UB_UINT8_LINE_SIZE;
    const uint32_t go_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t go32_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA;
    const uint32_t tv32_ubuf_offset = 10 * UB_UINT8_BLOCK_SIZE_MLA;
};



#endif

#ifdef __DAV_C220_CUBE__
template <TilingKeyType tilingKeyType = TilingKeyType::TILING_HALF_DATA, typename IN_DTYPE = half, typename IN_ROPE_DTYPE = half,  typename OUT_DTYPE = half, typename IN_KVDTYPE = half, InputFormat KInputType = InputFormat::ND_FORMAT, bool EnableOptimization = false>
class MLAttentionDecoderAic {
    // define dtype
    using mm1OutputType = typename AttentionType<tilingKeyType>::mm1OutputType;
    using mm1CopyType = typename AttentionType<tilingKeyType>::mm1CopyType;
    using mmBiasType = typename AttentionType<tilingKeyType>::mmBiasType;
    using mmScaleType = typename AttentionType<tilingKeyType>::mmScaleType;
    using mm2OutputType = typename AttentionType<tilingKeyType>::mm2OutputType;
    using mm2CopyType = typename AttentionType<tilingKeyType>::mm2CopyType;
    static constexpr uint32_t T_CUBE_MATRIX_SIZE = CUBE_MATRIX_SIZE_512 / sizeof(IN_DTYPE);
    static constexpr uint32_t T_BLOCK_SIZE =  BLOCK_SIZE_32 / sizeof(IN_DTYPE);
    static constexpr uint32_t T_BLOCK_OFFSET = 2 / sizeof(IN_DTYPE);
    static constexpr int32_t L1_KV_HALF_SIZE = 73728;// 2* 128 * 256
    static constexpr int32_t L1_KV_UINT8_SIZE = 73728 * 2;

public:
    __aicore__ __attribute__((always_inline)) inline MLAttentionDecoderAic() {
    }

    __aicore__ __attribute__((always_inline)) inline void SetArgs(
        __gm__ uint8_t *__restrict__ q_in_gm,
        __gm__ uint8_t *__restrict__ q_rope_in_gm,
        __gm__ uint8_t *__restrict__ k_in_gm,
        __gm__ uint8_t *__restrict__ k_rope_in_gm,
        __gm__ uint8_t *__restrict__ block_tables_in_gm,
        __gm__ uint8_t *__restrict__ o_out_gm,
        __gm__ uint8_t *__restrict__ s_out_gm,
        __gm__ uint8_t *__restrict__ s_rope_out_gm,
        __gm__ uint8_t *__restrict__ p_out_gm,
        __gm__ uint8_t *__restrict__ o_temp_gm,
        __gm__ uint8_t *__restrict__ tiling_para_gm)
    {
        SetPadding<uint64_t>(0);
        SetAtomicnone();
        SetNdpara(1, 0, 0);
        SetMasknorm();

        q_gm = reinterpret_cast<__gm__ IN_DTYPE *>(q_in_gm);
        q_rope_gm = reinterpret_cast<__gm__ IN_ROPE_DTYPE *>(q_rope_in_gm);
        k_gm = reinterpret_cast<__gm__ IN_KVDTYPE *>(k_in_gm);
        k_rope_gm = reinterpret_cast<__gm__ IN_ROPE_DTYPE *>(k_rope_in_gm);
        block_tables_gm = reinterpret_cast<__gm__ int32_t *>(block_tables_in_gm);
        s_gm = reinterpret_cast<__gm__ mm1CopyType *>(s_out_gm);

        p_gm = reinterpret_cast<__gm__ IN_DTYPE *>(p_out_gm);
        o_tmp_gm = reinterpret_cast<__gm__ mm2CopyType *>(o_temp_gm);
        tiling_gm = reinterpret_cast<__gm__ uint8_t *>(tiling_para_gm);

        q_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(q_in_gm));
        q_rope_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_ROPE_DTYPE *>(q_rope_gm));
        k_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_KVDTYPE *>(k_in_gm));
        k_rope_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_ROPE_DTYPE *>(k_rope_gm));
        s_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ mm1CopyType *>(s_out_gm));
        p_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(p_out_gm));
        o_tmp_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ mm2CopyType *>(o_temp_gm));
        block_tables_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_tables_in_gm));
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            s_rope_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(s_rope_out_gm));
            l1q_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(0);
            l1q_rope_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, IN_ROPE_DTYPE>(128*512*2);
            l1kv_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(128*576*2);
            l1kv_rope_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, IN_ROPE_DTYPE>(128 * 576 * 2 + 128 * 512 * 2);
            l1p_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(128 * 576 * 6);
        } else {
            l1q_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(l1q_buf_addr_offset);
            l1kv_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(l1kv_buf_addr_offset);
            l1p_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(l1p_buf_addr_offset);
        }

        num_batches = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm));
        q_heads = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_NUMHEADS));
        embedding_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADDIM));
        block_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_BLOCKSIZE));
        max_num_blocks_per_query = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MAXBLOCKS));
        kv_heads = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_KVHEADS));
        tiling_head_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADSIZE));
        tiling_para_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_PARASIZE));
        cur_qn_blk_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MTP_HEAD_SPLIT_SIZE));
        block_size_calc = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_BLOCKSIZE_CALC));
        mask_type = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MASK_TYPE_ND));
        totalTaskNum = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + 13));
        maxKVSeqLen = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MAX_KV_SEQ_LEN));

        num_batches_pad = RoundUp<16>(num_batches);

        stride_kv = static_cast<uint64_t>(kv_heads) * 512;
        stride_kv_rope = static_cast<uint64_t>(kv_heads) * 64;

        __k = embedding_size;
        round_k = RoundUp<T_BLOCK_SIZE>(__k);
        __v = embedding_size;
        stride_vo = static_cast<uint64_t>(kv_heads) * embedding_size;
        round_v = RoundUp<BLOCK_SIZE>(__v);
        embed_split_size_qk = 128;
        embed_split_loop_qk = (embedding_size + embed_split_size_qk - 1) / embed_split_size_qk;
    }


    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        SET_FLAG(M, MTE1, EVENT_ID2);
        SET_FLAG(M, MTE1, EVENT_ID3);
        SET_FLAG(M, MTE1, EVENT_ID4);
        SET_FLAG(M, MTE1, EVENT_ID5);
        SET_FLAG(M, MTE1, EVENT_ID6);
	    SET_FLAG(M, MTE1, EVENT_ID7);
        SET_FLAG(FIX, M, EVENT_ID0);
        SET_FLAG(FIX, M, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID0);
        SET_FLAG(MTE1, MTE2, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID2);
        SET_FLAG(MTE1, MTE2, EVENT_ID3);
        SET_FLAG(MTE1, MTE2, EVENT_ID4);
        SET_FLAG(MTE1, MTE2, EVENT_ID5);
        SET_FLAG(MTE1, MTE2, EVENT_ID6);
        SET_FLAG(MTE1, MTE2, EVENT_ID7);
        SET_FLAG(FIX, MTE1, EVENT_ID0);
        SET_FLAG(FIX, MTE1, EVENT_ID1);
        SET_FLAG(FIX, MTE1, EVENT_ID2);
        SET_FLAG(FIX, MTE1, EVENT_ID3);
        SET_FLAG(FIX, MTE1, EVENT_ID4);
        SET_FLAG(FIX, MTE1, EVENT_ID5);
        SET_FLAG(MTE2, FIX, EVENT_ID0);


        uint64_t cur_batch = 0;

        uint32_t q_block_num_per_batch = (q_heads + cur_qn_blk_size - 1) / cur_qn_blk_size;
        uint32_t process_num = q_block_num_per_batch * num_batches;

        for (uint32_t process = block_idx; process < process_num; process += (uint32_t)block_num) {  // for task
            cur_batch = process / q_block_num_per_batch;
            if (cur_batch >= num_batches) break;

            uint32_t offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
            uint32_t start_core_idx = (cur_batch * q_block_num_per_batch) % block_num;

            uint32_t q_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));
            uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 1 + offset_tiling));
            if (kv_seqlen == 0) {
                continue;
            }
            uint32_t kv_seqlen_align = (kv_seqlen + block_size - 1) / block_size * block_size;

            uint32_t start_head = (process % q_block_num_per_batch) * cur_qn_blk_size;
            uint32_t start_kv = 0;
            uint32_t cur_q_seq_len = q_seqlen;
            uint32_t cur_kv_seqlen = kv_seqlen;
            uint32_t cur_head_num = cur_qn_blk_size;

            InnerRunCubeMLA(cur_batch, start_head, cur_head_num, start_kv, cur_q_seq_len, cur_kv_seqlen,
                            offset_tiling);
        }
        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(M, MTE1, EVENT_ID2);
        WAIT_FLAG(M, MTE1, EVENT_ID3);
        WAIT_FLAG(M, MTE1, EVENT_ID4);
        WAIT_FLAG(M, MTE1, EVENT_ID5);
        WAIT_FLAG(M, MTE1, EVENT_ID6);
        WAIT_FLAG(M, MTE1, EVENT_ID7);
        WAIT_FLAG(FIX, M, EVENT_ID0);
        WAIT_FLAG(FIX, M, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID5);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID6);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
        WAIT_FLAG(FIX, MTE1, EVENT_ID0);
        WAIT_FLAG(FIX, MTE1, EVENT_ID1);
        WAIT_FLAG(FIX, MTE1, EVENT_ID2);
        WAIT_FLAG(FIX, MTE1, EVENT_ID3);
        WAIT_FLAG(FIX, MTE1, EVENT_ID4);
        WAIT_FLAG(FIX, MTE1, EVENT_ID5);
        WAIT_FLAG(MTE2, FIX, EVENT_ID0);
        PIPE_BARRIER(ALL);
    }

    __aicore__ __attribute__((always_inline)) inline void RunTP1()
    {
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        SET_FLAG(M, MTE1, EVENT_ID2);
        SET_FLAG(M, MTE1, EVENT_ID3);
        SET_FLAG(M, MTE1, EVENT_ID4);
        SET_FLAG(M, MTE1, EVENT_ID5);
        SET_FLAG(M, MTE1, EVENT_ID6);
	    SET_FLAG(M, MTE1, EVENT_ID7);
        SET_FLAG(FIX, M, EVENT_ID0);
        SET_FLAG(FIX, M, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID0);
        SET_FLAG(MTE1, MTE2, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID2);
        SET_FLAG(MTE1, MTE2, EVENT_ID3);
        SET_FLAG(MTE1, MTE2, EVENT_ID4);
        SET_FLAG(MTE1, MTE2, EVENT_ID5);
        SET_FLAG(MTE1, MTE2, EVENT_ID6);
        SET_FLAG(MTE1, MTE2, EVENT_ID7);
        SET_FLAG(FIX, MTE1, EVENT_ID0);
        SET_FLAG(FIX, MTE1, EVENT_ID1);
        SET_FLAG(FIX, MTE1, EVENT_ID2);
        SET_FLAG(FIX, MTE1, EVENT_ID3);
        SET_FLAG(FIX, MTE1, EVENT_ID4);
        SET_FLAG(FIX, MTE1, EVENT_ID5);
        SET_FLAG(MTE2, FIX, EVENT_ID0);

        uint32_t tail = totalTaskNum % block_num;
        if constexpr (EnableOptimization) {

        } else{
            tail = 0; // control whether to run tail optimization
        }
        uint32_t totalTaskNumRound = totalTaskNum - tail;
        
        
        for (uint32_t process = block_idx; process < totalTaskNumRound; process += (uint32_t)block_num) {  // for task
            uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
            uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));

            uint32_t q_seqlen = 1;
            uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
            if (kv_seqlen == 0) {
                continue;
            }
            uint32_t kv_seqlen_align = (kv_seqlen + block_size - 1) / block_size * block_size;

            uint32_t start_head = 0;
            uint32_t start_kv = 0;
            uint32_t cur_q_seq_len = q_seqlen;
            uint32_t cur_kv_seqlen = kv_seqlen;
            uint32_t cur_head_num = q_heads;

            InnerRunCubeMLATP1(cur_batch, start_head, cur_head_num, start_kv, cur_q_seq_len, cur_kv_seqlen, offset_tiling);
        }

        // suppose all seqs have same length
        if (tail > 0){
            uint32_t sample_kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + tiling_head_size + 2));
            bool enableExtraOptimization = true;
            if (block_num % 4 == 3) {
                // cannot optimize this situation due to math problem
                enableExtraOptimization = false;
            }
            if (sample_kv_seqlen <= 2048){
                // Too Short to benefit from optimization
                enableExtraOptimization = false;
            }
            if (!enableExtraOptimization || tail <= block_num / 2) {
                // collect all metadata
                uint32_t cores_per_seq = 1;
                if (0 < tail && tail <= block_num / 4) {// 6 tasks left, each works with 4 cores
                    cores_per_seq = 4;
                    if (tail == 1){
                        cores_per_seq = block_num;
                    }
                    else if (tail == 2){
                        cores_per_seq = block_num / 2;
                    }
                    else if(tail == 3){
                        cores_per_seq = block_num / 3;
                    }
                    else if(tail == 4){
                        cores_per_seq = block_num / 4;
                    }
                }
                else if(block_num / 4 < tail && tail <= block_num / 3) { // 8 tasks left, each works with 3 cores
                    cores_per_seq = 3;
    
                }
                else if(block_num / 3 < tail && tail <= block_num / 2) { // 12 tasks left, each works with 2 cores
                    cores_per_seq = 2;
                }
                else {
                    // no extra optimization for tail > 12
                    cores_per_seq = 1;
                }

                if(!enableExtraOptimization){
                    cores_per_seq = 1;
                }
    
                uint32_t process = totalTaskNumRound + block_idx / cores_per_seq;
                if (process < totalTaskNum) {
                    uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
                    uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));
        
                    uint32_t q_seqlen = 1;
                    uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
                    uint32_t kv_seqlen_each = kv_seqlen / cores_per_seq;
                    uint32_t kv_seqlen_align = (kv_seqlen_each + block_size - 1) / block_size * block_size;
                    uint32_t actual_work_cores = kv_seqlen / kv_seqlen_align + (kv_seqlen % kv_seqlen_align != 0);
                    // cores_per_seq = actual_work_cores;
                    uint32_t kv_seqlen_process = 0;
                    if (block_idx < block_idx / cores_per_seq * cores_per_seq + actual_work_cores){
                        kv_seqlen_process = (block_idx % cores_per_seq == actual_work_cores - 1) ? 
                            (kv_seqlen - kv_seqlen_align * (actual_work_cores - 1)) : kv_seqlen_align;
                    }
                        
                    if (kv_seqlen > 0 && kv_seqlen_process > 0) {
                        uint32_t start_head = 0;
                        uint32_t start_kv = (block_idx % cores_per_seq) * kv_seqlen_align;
                        uint32_t cur_q_seq_len = q_seqlen;
                        uint32_t cur_kv_seqlen = kv_seqlen_process;
                        uint32_t cur_head_num = q_heads;
            
                        // no need to modify anything in cube kernel, just call the same kernel
                        InnerRunCubeMLATP1(cur_batch, start_head, cur_head_num, start_kv, cur_q_seq_len, cur_kv_seqlen, offset_tiling);
                    }
                }
            }
            else if (tail > 3 * block_num / 4){
                // no benefit for optimizing this situation
                uint32_t process = totalTaskNumRound + block_idx;
                if (process < totalTaskNum) {
                    uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
                    uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));

                    uint32_t q_seqlen = 1;
                    uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
                    if (kv_seqlen > 0) {
                        uint32_t kv_seqlen_align = (kv_seqlen + block_size - 1) / block_size * block_size;
    
                        uint32_t start_head = 0;
                        uint32_t start_kv = 0;
                        uint32_t cur_q_seq_len = q_seqlen;
                        uint32_t cur_kv_seqlen = kv_seqlen;
                        uint32_t cur_head_num = q_heads;
    
                        InnerRunCubeMLATP1(cur_batch, start_head, cur_head_num, start_kv, cur_q_seq_len, cur_kv_seqlen, offset_tiling);
                    }
                }
            }
            else {
                // 18 >= tail >= 12 
                // first 12 tasks, two cores per task
                {
                    uint32_t cores_per_seq = 2;
                    uint32_t process = totalTaskNumRound + block_idx / cores_per_seq;
                    uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
                    uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));
        
                    uint32_t q_seqlen = 1;
                    uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
                    uint32_t kv_seqlen_each = kv_seqlen / cores_per_seq;
                    uint32_t kv_seqlen_align = (kv_seqlen_each + block_size - 1) / block_size * block_size;
                    uint32_t kv_seqlen_process = (block_idx % cores_per_seq == cores_per_seq - 1) ? 
                    (kv_seqlen - kv_seqlen_align * (cores_per_seq - 1)) : kv_seqlen_align;
                    
                    if (kv_seqlen > 0 && kv_seqlen_process > 0) {
                        uint32_t start_head = 0;
                        uint32_t start_kv = (block_idx % cores_per_seq) * kv_seqlen_align;
                        uint32_t cur_q_seq_len = q_seqlen;
                        uint32_t cur_kv_seqlen = kv_seqlen_process;
                        uint32_t cur_head_num = q_heads;
            
                        // no need to modify anything in cube kernel, just call the same kernel
                        InnerRunCubeMLATP1(cur_batch, start_head, cur_head_num, start_kv, cur_q_seq_len, cur_kv_seqlen, offset_tiling);
                    }
                }
                {
                    uint32_t cores_per_seq = 4;
                    uint32_t process = totalTaskNumRound + block_num / 2 + block_idx / cores_per_seq;
                    if (process < totalTaskNum) {
                        uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
                        uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));
            
                        uint32_t q_seqlen = 1;
                        uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
                        uint32_t kv_seqlen_each = kv_seqlen / cores_per_seq;
                        uint32_t kv_seqlen_align = (kv_seqlen_each + block_size - 1) / block_size * block_size;
                        uint32_t kv_seqlen_process = (block_idx % cores_per_seq == cores_per_seq - 1) ? 
                        (kv_seqlen - kv_seqlen_align * (cores_per_seq - 1)) : kv_seqlen_align;
                        
                        if (kv_seqlen > 0 && kv_seqlen_process > 0) {
                            uint32_t start_head = 0;
                            uint32_t start_kv = (block_idx % cores_per_seq) * kv_seqlen_align;
                            uint32_t cur_q_seq_len = q_seqlen;
                            uint32_t cur_kv_seqlen = kv_seqlen_process;
                            uint32_t cur_head_num = q_heads;
                
                            // no need to modify anything in cube kernel, just call the same kernel
                            InnerRunCubeMLATP1(cur_batch, start_head, cur_head_num, start_kv, cur_q_seq_len, cur_kv_seqlen, offset_tiling);
                        }
                    }
                }
            }
        }

        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(M, MTE1, EVENT_ID2);
        WAIT_FLAG(M, MTE1, EVENT_ID3);
        WAIT_FLAG(M, MTE1, EVENT_ID4);
        WAIT_FLAG(M, MTE1, EVENT_ID5);
        WAIT_FLAG(M, MTE1, EVENT_ID6);
        WAIT_FLAG(M, MTE1, EVENT_ID7);
        WAIT_FLAG(FIX, M, EVENT_ID0);
        WAIT_FLAG(FIX, M, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID5);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID6);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
        WAIT_FLAG(FIX, MTE1, EVENT_ID0);
        WAIT_FLAG(FIX, MTE1, EVENT_ID1);
        WAIT_FLAG(FIX, MTE1, EVENT_ID2);
        WAIT_FLAG(FIX, MTE1, EVENT_ID3);
        WAIT_FLAG(FIX, MTE1, EVENT_ID4);
        WAIT_FLAG(FIX, MTE1, EVENT_ID5);
        WAIT_FLAG(MTE2, FIX, EVENT_ID0);
        PIPE_BARRIER(ALL);
    }

private:
    __aicore__ __attribute__((always_inline)) inline void InnerRunCubeMLA(uint32_t cur_batch, uint32_t start_head, uint32_t cur_head_num,
        uint32_t start_kv, uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t offset_tiling)
    {
        uint32_t addr_q_high32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
        uint32_t addr_q_loww32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 3 + offset_tiling));
        uint64_t addr_q_scalar = (uint64_t)(((uint64_t)addr_q_high32) << 32 | addr_q_loww32);
        uint64_t q_offset = addr_q_scalar * 512 + start_head * 512;
        uint64_t q_rope_offset = addr_q_scalar * 64 + start_head * 64;

        uint32_t pp_n_scalar = block_size;
        uint32_t sub_n_loop = pp_n_scalar / block_size;

        uint32_t n_loop = (cur_kv_seqlen + pp_n_scalar - 1) / pp_n_scalar;

        uint32_t qk_n = pp_n_scalar;
        uint32_t qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);
        uint32_t qk_n_2 = pp_n_scalar;
        uint32_t qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);
        uint32_t qk_round_n_l1 = RoundUp<T_BLOCK_SIZE>(qk_n);
        uint32_t qk_round_n_2_l1 = RoundUp<T_BLOCK_SIZE>(qk_n_2);
        uint64_t hidden_size = 576;
        if constexpr(tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            hidden_size = 512;
        }
        uint64_t k_round_n = qk_round_n;
        uint32_t row_num  = cur_head_num * cur_q_seqlen;
        m = RoundUp<16>(row_num);

        // copy Q
        if (cur_q_seqlen == 1) {
            gm_to_l1<ArchType::ASCEND_V220, IN_DTYPE, DataFormat::ND, DataFormat::NZ>(
                l1q_buf_addr_tensor,
                q_gm_tensor[q_offset],
                cur_head_num,        // nValue
                RoundUp<16>(cur_head_num),// dstNzC0Stride
                0,                     // dstNzMatrixStride, unused
                512,                   // dValue
                0,                     // dstNzMatrixStride, unused
                512                   // srcDValue
            );
        } else {
            if (q_heads < 128) {
                AscendC::DataCopy(
                    l1q_buf_addr_tensor,
                    q_gm_tensor[q_offset],
                    AscendC::Nd2NzParams(
                        cur_q_seqlen,                // ndNum
                        cur_head_num,                 // nValue
                        512,                            // dValue
                        512 * q_heads,        // srcNdMatrixStride
                        512,    // srcDValue
                        RoundUp<16>(cur_head_num * cur_q_seqlen), // dstNzC0Stride
                        cur_q_seqlen,                   // dstNzNStride
                        16             // dstNzMatrixStride
                    )
                );
            } else {
                for (uint32_t ii =0; ii < cur_q_seqlen; ii++) {
                    AscendC::DataCopy(
                        l1q_buf_addr_tensor[ii * 16], // offset one datablock
                        q_gm_tensor[q_offset + ii * q_heads * 512],
                        AscendC::Nd2NzParams(
                            1,                // ndNum
                            cur_head_num,                 // nValue
                            512,                            // dValue
                            0,        // srcNdMatrixStride
                            512,    // srcDValue
                            RoundUp<16>(cur_q_seqlen * cur_head_num), // dstNzC0Stride
                            cur_q_seqlen,                   // dstNzNStride
                            16             // dstNzMatrixStride
                        )
                    );
                }
            }

        }
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            gm_to_l1<ArchType::ASCEND_V220, IN_ROPE_DTYPE, DataFormat::ND, DataFormat::NZ>(
                l1q_rope_buf_addr_tensor,
                q_rope_gm_tensor[q_rope_offset],
                cur_head_num,        // nValue
                RoundUp<16>(cur_head_num),// dstNzC0Stride
                0,                     // dstNzMatrixStride, unused
                64,                   // dValue
                0,                     // dstNzMatrixStride, unused
                64                   // srcDValue
            );
        } else {
            AscendC::DataCopy(
                l1q_buf_addr_tensor[RoundUp<16>(cur_head_num * cur_q_seqlen) * 512],
                q_rope_gm_tensor[q_rope_offset],
                AscendC::Nd2NzParams(
                    cur_head_num,                // ndNum, 32
                    cur_q_seqlen,                 // nValue, 4
                    64,                            // dValue
                    64,                 // srcNdMatrixStride
                    64 * q_heads,                            // srcDValue
                    RoundUp<16>(cur_head_num * cur_q_seqlen),    // dstNzC0Stride
                    1,                              // dstNzNStride
                    16 * cur_q_seqlen             // dstNzMatrixStride
                )
            );
        }
        SET_FLAG(MTE2, MTE1, EVENT_ID0);
        WAIT_FLAG(MTE2, MTE1, EVENT_ID0);
        for (uint32_t n_idx = 0; n_idx < n_loop + 1; n_idx+=1) {
            if (n_idx != n_loop) {
                uint32_t l1_kv_pingpong_flag = n_idx % 2;
                if (n_idx == (n_loop - 1)) {
                    qk_n = (cur_kv_seqlen - n_idx * pp_n_scalar);
                    qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);
                    qk_round_n_l1 = RoundUp<T_BLOCK_SIZE>(qk_n);
                }
                if constexpr(tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
                    k_round_n = qk_round_n_l1;
                } else {
                    k_round_n = qk_round_n;
                }
                uint64_t hiddenSize_offset = start_head * cur_q_seqlen * embedding_size;
                uint32_t embed_split_size = 128;
                uint32_t round_embed_split_size = RoundUp<T_BLOCK_SIZE>(embed_split_size);

                /* ************ CUBE1 stage1  ************* */

                uint32_t block_table_id = (uint32_t)(*(block_tables_gm +
                                cur_batch * max_num_blocks_per_query + start_kv / block_size + n_idx));
                int64_t kv_offset = (int64_t)block_table_id * block_size * stride_kv;
                int64_t kv_offset_rope = (int64_t)block_table_id * block_size * stride_kv_rope;
                uint32_t q_load_coeff = 1;
                q_load_coeff = m;
                WAIT_FLAG(MTE1, MTE2, l1_kv_pingpong_flag);  // wait for v -> L0B
                if constexpr(KInputType == InputFormat::ND_FORMAT) {
                    gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::ND, DataFormat::NZ>(
                        l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 576],
                        k_gm_tensor[kv_offset],
                        qk_n,         // nValue
                        qk_round_n,             // dstNzC0Stride
                        0,                     // dstNzMatrixStride, unused
                        512,            // dValue
                        0,                     // dstNzMatrixStride, unused
                        stride_kv            // srcDValue
                    );
                    SET_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                    WAIT_FLAG(MTE1, MTE2, l1_kv_pingpong_flag + 2);  // wait for v -> L0B
                    gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::ND, DataFormat::NZ>(
                        l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 576 + 512 * qk_round_n],
                        k_rope_gm_tensor[kv_offset_rope],
                        qk_n,         // nValue
                        qk_round_n,             // dstNzC0Stride
                        0,                     // dstNzMatrixStride, unused
                        64,            // dValue
                        0,                     // dstNzMatrixStride, unused
                        stride_kv_rope            // srcDValue
                    );
                } else if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
                    gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::NZ, DataFormat::NZ>(
                        l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 512],
                        k_gm_tensor[kv_offset],
                        qk_round_n_l1,         // nValue
                        block_size,         // dstNzC0Stride
                        0,                     // dstNzMatrixStride, unused
                        512,            // dValue
                        0,                     // dstNzMatrixStride, unused
                        0            // srcDValue
                    );
                    SET_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                    WAIT_FLAG(MTE1, MTE2, l1_kv_pingpong_flag + 2);  // wait for v -> L0B
                    gm_to_l1<ArchType::ASCEND_V220, IN_ROPE_DTYPE, DataFormat::NZ, DataFormat::NZ>(
                        l1kv_rope_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 64],
                        k_rope_gm_tensor[kv_offset_rope],
                        qk_round_n,         // nValue
                        block_size,             // dstNzC0Stride
                        0,                     // dstNzMatrixStride, unused
                        64,            // dValue
                        0,                     // dstNzMatrixStride, unused
                        0            // srcDValue
                    );
                } else {
                    gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::NZ, DataFormat::NZ>(
                        l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 576],
                        k_gm_tensor[kv_offset],
                        qk_round_n,         // nValue
                        block_size,             // dstNzC0Stride
                        0,                     // dstNzMatrixStride, unused
                        512,            // dValue
                        0,                     // dstNzMatrixStride, unused
                        0            // srcDValue
                    );

                    SET_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                    WAIT_FLAG(MTE1, MTE2, l1_kv_pingpong_flag + 2);  // wait for v -> L0B
                    gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::NZ, DataFormat::NZ>(
                        l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 576 + 512 * qk_round_n],
                        k_rope_gm_tensor[kv_offset_rope],
                        qk_round_n,         // nValue
                        block_size,             // dstNzC0Stride
                        0,                     // dstNzMatrixStride, unused
                        64,            // dValue
                        0,                     // dstNzMatrixStride, unused
                        0            // srcDValue
                    );
                }

                SET_FLAG(MTE2, MTE1, l1_kv_pingpong_flag + 2);
                uint64_t hidden_split_time = (hidden_size + 128 - 1) / 128;
                uint64_t embed_split_idx = 0;
                for (embed_split_idx = 0; embed_split_idx < hidden_split_time; ++embed_split_idx) {
                    if (embed_split_idx == 4) {
                        embed_split_size = 64;
                        round_embed_split_size = 64;
                    }
                    WAIT_FLAG(M, MTE1, embed_split_idx % 2);

                    for (uint64_t loa_load_idx = 0; loa_load_idx < q_load_coeff / BLOCK_SIZE; ++loa_load_idx) {
                        l1_to_l0_a<ArchType::ASCEND_V220, IN_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                            l0a_buf_tensor[embed_split_idx % 2 * 16384 + loa_load_idx * round_embed_split_size * BLOCK_SIZE],
                            l1q_buf_addr_tensor[embed_split_idx * m * 128 + loa_load_idx * T_CUBE_MATRIX_SIZE],
                            0,
                            round_embed_split_size / T_BLOCK_SIZE,                                 // repeat
                            0,
                            q_load_coeff / BLOCK_SIZE,                            // srcStride
                            0,
                            0                                                     // dstStride
                        );
                    }

                    SET_FLAG(MTE1, M, embed_split_idx % 2);

                    if (embed_split_idx == 0) {
                        WAIT_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                    }
                    if (embed_split_idx == 4) {
                        WAIT_FLAG(MTE2, MTE1, l1_kv_pingpong_flag + 2);
                    }
                    WAIT_FLAG(M, MTE1, embed_split_idx % 2 + 2);
                    l1_to_l0_b<ArchType::ASCEND_V220, IN_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                        l0b_buf_tensor[embed_split_idx % 2 * 16384],
                        l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * hidden_size + embed_split_idx * k_round_n * 128],
                        0,
                        round_embed_split_size * k_round_n / T_CUBE_MATRIX_SIZE,  // repeat
                        0,
                        1,                                        // srcStride
                        0,
                        0                                        // dstStride
                    );
                    if (embed_split_idx == 4) {
                        SET_FLAG(MTE1, MTE2, l1_kv_pingpong_flag + 2);
                    }
                    SET_FLAG(MTE1, M, embed_split_idx % 2 + 2);
                    WAIT_FLAG(MTE1, M, embed_split_idx % 2);
                    WAIT_FLAG(MTE1, M, embed_split_idx % 2 + 2);
                    if (embed_split_idx == 0) {
                        WAIT_FLAG(FIX, M, l1_kv_pingpong_flag);
                    }
                    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
                        mmad<ArchType::ASCEND_V220, IN_DTYPE, IN_DTYPE, mm1OutputType, false>(
                            mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
                            l0a_buf_tensor[embed_split_idx % 2 * 16384],
                            l0b_buf_tensor[embed_split_idx % 2 * 16384],
                            m,     // m
                            qk_round_n_l1,  // n
                            embed_split_size,   // k
                            embed_split_idx == 0     // cmatrixInitVal
                        );
                    } else {
                        mmad<ArchType::ASCEND_V220, IN_DTYPE, IN_DTYPE, mm1OutputType, false>(
                            mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
                            l0a_buf_tensor[embed_split_idx % 2 * 16384],
                            l0b_buf_tensor[embed_split_idx % 2 * 16384],
                            m,     // m
                            qk_n,  // n
                            embed_split_size,   // k
                            embed_split_idx == 0     // cmatrixInitVal
                        );
                    }

                    PIPE_BARRIER(M);
                    SET_FLAG(M, MTE1, embed_split_idx % 2);
                    SET_FLAG(M, MTE1, embed_split_idx % 2 + 2);

                    // copy S to gm
                    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
                        if (embed_split_idx == 3) {
                            SET_FLAG(M, FIX, l1_kv_pingpong_flag);
                            WAIT_FLAG(M, FIX, l1_kv_pingpong_flag);

                            l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, mm1CopyType, mm1OutputType>(
                                s_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER + (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
                                mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
                                m,           // MSize
                                qk_n,  // NSize
                                RoundUp<16>(m), // srcStride
                                qk_round_n  // dstStride_dst_D
                            );
                            SET_FLAG(FIX, M, l1_kv_pingpong_flag);
                        }
                    }
                    if (embed_split_idx == 4) {
                        SET_FLAG(M, FIX, l1_kv_pingpong_flag);
                        WAIT_FLAG(M, FIX, l1_kv_pingpong_flag);

                        l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, mm1CopyType, mm1OutputType>(
                            s_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER + (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
                            mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
                            m,           // MSize
                            qk_round_n,  // NSize
                            RoundUp<16>(m), // srcStride
                            qk_round_n  // dstStride_dst_D
                        );
                        SET_FLAG(FIX, M, l1_kv_pingpong_flag);
                    }
                }
                if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
                    embed_split_idx = 4;
                    embed_split_size = 64;
                    round_embed_split_size = 64;
                    WAIT_FLAG(M, MTE1, embed_split_idx % 2);

                    for (uint64_t loa_load_idx = 0; loa_load_idx < q_load_coeff / BLOCK_SIZE; ++loa_load_idx) {
                        l1_to_l0_a<ArchType::ASCEND_V220, IN_ROPE_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                            l0a_buf_tensor.template ReinterpretCast<IN_ROPE_DTYPE>()[embed_split_idx % 2 * 16384 * 2 + loa_load_idx * round_embed_split_size * BLOCK_SIZE],
                            l1q_rope_buf_addr_tensor[loa_load_idx * CUBE_MATRIX_SIZE],
                            0,
                            round_embed_split_size / BLOCK_SIZE,                                 // repeat
                            0,
                            q_load_coeff / BLOCK_SIZE,                            // srcStride
                            0,
                            0                                                     // dstStride
                        );
                    }

                    SET_FLAG(MTE1, M, embed_split_idx % 2);

                    WAIT_FLAG(MTE2, MTE1, l1_kv_pingpong_flag + 2);
                    WAIT_FLAG(M, MTE1, embed_split_idx % 2 + 2);
                    l1_to_l0_b<ArchType::ASCEND_V220, IN_ROPE_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                        l0b_buf_tensor.template ReinterpretCast<IN_ROPE_DTYPE>()[embed_split_idx % 2 * 16384 * 2],
                        l1kv_rope_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 64],
                        0,
                        round_embed_split_size * qk_round_n / CUBE_MATRIX_SIZE,  // repeat
                        0,
                        1,                                        // srcStride
                        0,
                        0                                        // dstStride
                    );

                    SET_FLAG(MTE1, MTE2, l1_kv_pingpong_flag + 2);
                    SET_FLAG(MTE1, M, embed_split_idx % 2 + 2);
                    WAIT_FLAG(MTE1, M, embed_split_idx % 2);
                    WAIT_FLAG(MTE1, M, embed_split_idx % 2 + 2);
                    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
                        WAIT_FLAG(FIX, M, l1_kv_pingpong_flag);
                    }
                    mmad<ArchType::ASCEND_V220, IN_ROPE_DTYPE, IN_ROPE_DTYPE, float, false>(
                        mm1_l0c_buf_tensor.template ReinterpretCast<float>()[l1_kv_pingpong_flag * 16384],
                        l0a_buf_tensor.template ReinterpretCast<IN_ROPE_DTYPE>()[embed_split_idx % 2 * 16384 * 2],
                        l0b_buf_tensor.template ReinterpretCast<IN_ROPE_DTYPE>()[embed_split_idx % 2 * 16384 * 2],
                        m,     // m
                        qk_n,  // n
                        embed_split_size,   // k
                        1     // cmatrixInitVal
                    );
                    PIPE_BARRIER(M);
                    SET_FLAG(M, MTE1, embed_split_idx % 2);
                    SET_FLAG(M, MTE1, embed_split_idx % 2 + 2);

                    SET_FLAG(M, FIX, l1_kv_pingpong_flag);
                    WAIT_FLAG(M, FIX, l1_kv_pingpong_flag);
                    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
                        l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, float, float>(
                            s_rope_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER + (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
                            mm1_l0c_buf_tensor.template ReinterpretCast<float>()[l1_kv_pingpong_flag * 16384],
                            m,           // MSize
                            qk_round_n,  // NSize
                            RoundUp<16>(m), // srcStride
                            qk_round_n  // dstStride_dst_D
                        );
                    } else {
                        l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, mm1CopyType, mm1OutputType>(
                            s_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER + (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
                            mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
                            m,           // MSize
                            qk_round_n,  // NSize
                            RoundUp<16>(m), // srcStride
                            qk_round_n  // dstStride_dst_D
                        );
                    }
                    SET_FLAG(FIX, M, l1_kv_pingpong_flag);
                }
                FftsCrossCoreSync<PIPE_FIX, 2>(QK_READY_DECODER);
            }
            /* ************ CUBE2 stage1  ************* */
            if (n_idx != 0) {
                if (n_idx == n_loop) {
                    qk_n_2 = (cur_kv_seqlen - (n_idx - 1) * pp_n_scalar);
                    qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);
                    qk_round_n_2_l1 = RoundUp<T_BLOCK_SIZE>(qk_n_2);
                }
                k_round_n = qk_round_n_2_l1;
                uint32_t l1_kv_pingpong_flag = (n_idx - 1) % 2;
                uint32_t l0_p_pingpong_flag = (n_idx - 1) % 2;
                uint32_t embed_split_size = 128;
                embed_split_loop_v = 4;
                uint32_t round_embed_split_size = RoundUp<T_BLOCK_SIZE>(embed_split_size);
                for (uint32_t embed_split_idx = 0; embed_split_idx < embed_split_loop_v; ++embed_split_idx) {
                    uint32_t l0c_pingpong_flag = (n_idx + embed_split_idx) % 2;
                    uint32_t l0b_pingpong_flag = (embed_split_idx + 1) % 2;
                    uint64_t l1kv_offset = embed_split_idx * k_round_n * round_embed_split_size;
                    WAIT_FLAG(M, MTE1, l0b_pingpong_flag + 2);
                    AscendC::LoadData2dTransposeParams loadDataParams;
                    loadDataParams.dstGap = 0;
                    loadDataParams.startIndex = 0;
                    loadDataParams.dstFracGap = 0;
                    if (k_round_n <= round_embed_split_size) { // Nz -> nZ
                        loadDataParams.repeatTimes = round_embed_split_size / T_BLOCK_SIZE;
                        loadDataParams.srcStride = k_round_n / T_BLOCK_SIZE;
                        uint16_t dstGap = sizeof(IN_DTYPE) == 1 ? 1 : 0;
                        loadDataParams.dstGap = dstGap;
                        for (uint32_t l0b_load_idx = 0; l0b_load_idx < k_round_n / T_BLOCK_SIZE; ++l0b_load_idx) {
                            // along embd dim
                            AscendC::LoadDataWithTranspose(
                                    l0b_buf_tensor[l0b_pingpong_flag * 16384 + l0b_load_idx * RoundUp<16>(embed_split_size) * T_BLOCK_SIZE],
                                    l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * hidden_size + l1kv_offset + l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE],
                                    loadDataParams);
                        }
                    } else {
                        for (uint32_t l0b_load_idx = 0; l0b_load_idx < round_embed_split_size / T_BLOCK_SIZE; ++l0b_load_idx) {
                            // along kv_len_blk dim
                            loadDataParams.repeatTimes = qk_round_n_2 / T_BLOCK_SIZE;
                            loadDataParams.srcStride = 1;
                            loadDataParams.dstGap = round_embed_split_size / BLOCK_SIZE - 1;
                            AscendC::LoadDataWithTranspose(
                                l0b_buf_tensor[l0b_pingpong_flag * 16384 + l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE],
                                l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * hidden_size + l1kv_offset + l0b_load_idx * qk_round_n_2 * T_BLOCK_SIZE],
                                loadDataParams);
                        }
                    }
                    if (embed_split_idx == embed_split_loop_v - 1) {
                        SET_FLAG(MTE1, MTE2, l1_kv_pingpong_flag);
                    }
                    // move p from gm to l1
                    uint32_t p_move_head_num = row_num;
                    if (embed_split_idx == 0) {
                        WaitFlagDev(SOFTMAX_READY_DECODER);

                        WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
                        gm_to_l1<ArchType::ASCEND_V220, IN_DTYPE, DataFormat::ND, DataFormat::NZ>(
                            l1p_buf_addr_tensor,
                            p_gm_tensor[(uint64_t)block_idx * TMP_SIZE * T_BLOCK_OFFSET + ((n_idx - 1) % 2) * TMP_SIZE * T_BLOCK_OFFSET / 2],
                            p_move_head_num,         // nValue
                            RoundUp<BLOCK_SIZE>(p_move_head_num),// dstNzC0Stride
                            0,                     // dstNzMatrixStride, unused
                            k_round_n,           // dValue
                            0,                     // dstNzMatrixStride, unused
                            qk_round_n_2 * 2 / sizeof(IN_DTYPE)           // srcDValue
                        );
                        SET_FLAG(MTE2, MTE1, EVENT_ID7);
                        WAIT_FLAG(MTE2, MTE1, EVENT_ID7);
                        // move p from l1 to l0a
                        WAIT_FLAG(M, MTE1, l0_p_pingpong_flag);
                        uint32_t p_load_coeff = RoundUp<16>(p_move_head_num);
                        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
                            l1_to_l0_a<ArchType::ASCEND_V220, IN_DTYPE, false, DataFormat::NZ, DataFormat::ZZ>(
                                l0a_buf_tensor[l0_p_pingpong_flag * 16384], l1p_buf_addr_tensor, RoundUp<BLOCK_SIZE>(p_move_head_num),
                                qk_round_n_2_l1, // repeat
                                0,
                                0, // srcStride
                                0,
                                0 // dstStride
                            );
                        } else {
                            for (uint64_t loa_load_idx = 0; loa_load_idx < p_load_coeff / BLOCK_SIZE; ++loa_load_idx) {
                                l1_to_l0_a<ArchType::ASCEND_V220, IN_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                                    l0a_buf_tensor[l0_p_pingpong_flag * 16384 + loa_load_idx * qk_round_n_2 * BLOCK_SIZE],
                                    l1p_buf_addr_tensor[loa_load_idx * T_CUBE_MATRIX_SIZE],
                                    0,
                                    qk_round_n_2 / T_BLOCK_SIZE,                                 // repeat
                                    0,
                                    p_load_coeff / BLOCK_SIZE,                               // srcStride
                                    0,
                                    0                                                        // dstStride
                                );
                            }
                        }
                        SET_FLAG(MTE1, MTE2, EVENT_ID7);
                    }
                    SET_FLAG(MTE1, M, l0b_pingpong_flag);
                    WAIT_FLAG(MTE1, M, l0b_pingpong_flag);
                    WAIT_FLAG(FIX, M, l0c_pingpong_flag);
                    mmad<ArchType::ASCEND_V220, IN_DTYPE, IN_DTYPE, mm2OutputType, false>(
                        mm2_l0c_buf_tensor[l0c_pingpong_flag * 16384],
                        l0a_buf_tensor[l0_p_pingpong_flag * 16384],
                        l0b_buf_tensor[l0b_pingpong_flag * 16384],
                        m,     // m
                        embed_split_size,   // n
                        qk_n_2,  // k
                        1      // cmatrixInitVal
                    );
                    SET_FLAG(M, MTE1, l0b_pingpong_flag + 2);
                    if (embed_split_idx == embed_split_loop_v - 1) {
                        SET_FLAG(M, MTE1, l0_p_pingpong_flag);
                    }
                    SET_FLAG(M, FIX, l0c_pingpong_flag);
                    WAIT_FLAG(M, FIX, l0c_pingpong_flag);

                    // copy O to gm
                    l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, mm2CopyType, mm2OutputType>(
                        o_tmp_gm_tensor[(uint64_t)block_idx * TMP_SIZE * 2 + embed_split_idx * round_embed_split_size + ((n_idx - 1) % 2) * TMP_SIZE],
                        mm2_l0c_buf_tensor[l0c_pingpong_flag * 16384],
                        m,        // MSize
                        RoundUp<16>(embed_split_size),  // NSize 32B align
                        RoundUp<16>(m),       // srcStride
                        round_v  // dstStride_dst_D
                    );
                    SET_FLAG(FIX, M, l0c_pingpong_flag);
                }
                FftsCrossCoreSync<PIPE_FIX, 2>(UPDATE_READY_DECODER);
            }
        }
    }

    __aicore__ __attribute__((always_inline)) inline void InnerRunCubeMLATP1(uint32_t cur_batch, uint32_t start_head, uint32_t cur_head_num,
        uint32_t start_kv, uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t offset_tiling)
    {
        uint32_t prev_task = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 1 + offset_tiling));
        uint64_t addr_q_scalar = (uint64_t)prev_task * q_heads;
        uint64_t q_offset = addr_q_scalar * 512 + start_head * 512;
        uint64_t q_rope_offset = addr_q_scalar * 64 + start_head * 64;

        uint32_t pp_n_scalar = block_size;
        uint32_t sub_n_loop = pp_n_scalar / block_size;

        uint32_t n_loop = (cur_kv_seqlen + pp_n_scalar - 1) / pp_n_scalar;

        uint32_t qk_n = pp_n_scalar;
        uint32_t qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);
        uint32_t qk_n_2 = pp_n_scalar;
        uint32_t qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);

        uint32_t row_num  = cur_head_num * cur_q_seqlen;

        uint32_t sv_n = n_loop == 1 ? cur_kv_seqlen : pp_n_scalar;
        m = RoundUp<16>(row_num);

        // copy Q
        if (cur_q_seqlen == 1) {
            gm_to_l1<ArchType::ASCEND_V220, IN_DTYPE, DataFormat::ND, DataFormat::NZ>(
                l1q_buf_addr_tensor,
                q_gm_tensor[q_offset],
                cur_head_num,        // nValue
                RoundUp<16>(cur_head_num),// dstNzC0Stride
                0,                     // dstNzMatrixStride, unused
                512,                   // dValue
                0,                     // dstNzMatrixStride, unused
                512                   // srcDValue
            );
        } else {
            if (q_heads < 128) {
                AscendC::DataCopy(
                    l1q_buf_addr_tensor,
                    q_gm_tensor[q_offset],
                    AscendC::Nd2NzParams(
                        cur_q_seqlen,                // ndNum
                        cur_head_num,                 // nValue
                        512,                            // dValue
                        512 * q_heads,        // srcNdMatrixStride
                        512,    // srcDValue
                        RoundUp<16>(cur_head_num * cur_q_seqlen), // dstNzC0Stride
                        cur_q_seqlen,                   // dstNzNStride
                        16             // dstNzMatrixStride
                    )
                );
            } else {
                for (uint32_t ii =0; ii < cur_q_seqlen; ii++) {
                    AscendC::DataCopy(
                        l1q_buf_addr_tensor[ii * 16], // offset one datablock
                        q_gm_tensor[q_offset + ii * q_heads * 512],
                        AscendC::Nd2NzParams(
                            1,                // ndNum
                            cur_head_num,                 // nValue
                            512,                            // dValue
                            0,        // srcNdMatrixStride
                            512,    // srcDValue
                            RoundUp<16>(cur_q_seqlen * cur_head_num), // dstNzC0Stride
                            cur_q_seqlen,                   // dstNzNStride
                            16             // dstNzMatrixStride
                        )
                    );
                }
            }

        }

        AscendC::DataCopy(
            l1q_buf_addr_tensor[RoundUp<16>(cur_head_num * cur_q_seqlen) * 512],
            q_rope_gm_tensor[q_rope_offset],
            AscendC::Nd2NzParams(
                cur_head_num,                // ndNum, 32
                cur_q_seqlen,                 // nValue, 4
                64,                            // dValue
                64,                 // srcNdMatrixStride
                64 * q_heads,                            // srcDValue
                RoundUp<16>(cur_head_num * cur_q_seqlen),    // dstNzC0Stride
                1,                              // dstNzNStride
                16 * cur_q_seqlen             // dstNzMatrixStride
            )
        );

        SET_FLAG(MTE2, MTE1, EVENT_ID0);
        WAIT_FLAG(MTE2, MTE1, EVENT_ID0);
        uint32_t s_block_stack = 4;
        for (uint32_t n_idx = 0; n_idx < n_loop + s_block_stack; n_idx+=s_block_stack) {
            if (n_idx < n_loop) {
                uint32_t sv_n_triu = n_loop * pp_n_scalar;
                if (n_idx + s_block_stack > n_loop - 1) {
                    sv_n = cur_kv_seqlen - n_idx * pp_n_scalar; // delete
                } else {
                    sv_n = pp_n_scalar * s_block_stack;
                }
                uint32_t sv_round_n = (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                for (uint32_t split_idx = 0; split_idx < s_block_stack && n_idx + split_idx < n_loop; split_idx++) {
                    uint32_t now_idx = n_idx + split_idx;
                    uint32_t l1_kv_pingpong_flag = now_idx % 2;
                    if (now_idx == (n_loop - 1)) {
                        qk_n = (cur_kv_seqlen - now_idx * pp_n_scalar);
                        qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);
                    } else  {
                        qk_n = pp_n_scalar;
                        qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);
                    }
                    bool last_split = split_idx == s_block_stack - 1 || now_idx == n_loop - 1;
                    uint32_t embed_split_size = 128;
                    uint32_t round_embed_split_size = RoundUp<T_BLOCK_SIZE>(embed_split_size);

                    /* ************ CUBE1 stage1  ************* */

                    uint32_t block_table_id = (uint32_t)(*(block_tables_gm +
                                    cur_batch * max_num_blocks_per_query + start_kv / block_size + now_idx));
                    int64_t kv_offset = (int64_t)block_table_id * block_size * stride_kv;
                    int64_t kv_offset_rope = (int64_t)block_table_id * block_size * stride_kv_rope;


                    uint32_t q_load_coeff = 1;
                    q_load_coeff = m;
                    int64_t now_l1_offset = 0;
                    for (uint32_t embed_split_idx = 0; embed_split_idx < 5; ++embed_split_idx) {
                        if (embed_split_idx == 4) {
                            embed_split_size = 64;
                            round_embed_split_size = 64;
                        }
                        WAIT_FLAG(M, MTE1, embed_split_idx % 2);

                        for (uint64_t loa_load_idx = 0; loa_load_idx < q_load_coeff / BLOCK_SIZE; ++loa_load_idx) {
                            l1_to_l0_a<ArchType::ASCEND_V220, IN_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                                l0a_buf_tensor[embed_split_idx % 2 * 16384 + loa_load_idx * round_embed_split_size * BLOCK_SIZE],
                                l1q_buf_addr_tensor[embed_split_idx * m * 128 + loa_load_idx * T_CUBE_MATRIX_SIZE],
                                0,
                                round_embed_split_size / T_BLOCK_SIZE,                                 // repeat
                                0,
                                q_load_coeff / BLOCK_SIZE,                            // srcStride
                                0,
                                0                                                     // dstStride
                            );
                        }

                        SET_FLAG(MTE1, M, embed_split_idx % 2);
                        if (embed_split_idx == 0 || embed_split_idx == 2) {
                            WAIT_FLAG(MTE1, MTE2, l1_kv_pingpong_flag);  // 等待V全部搬入L0B
                            now_l1_offset = l1_kv_pingpong_flag * 128 * 256;
                            if constexpr(KInputType == InputFormat::ND_FORMAT) {
                                gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::ND, DataFormat::NZ>(
                                    l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 256],
                                    k_gm_tensor[kv_offset + embed_split_idx * 128],
                                    qk_n,         // nValue
                                    qk_round_n,             // dstNzC0Stride
                                    0,                     // dstNzMatrixStride, unused
                                    256,            // dValue
                                    0,                     // dstNzMatrixStride, unused
                                    stride_kv            // srcDValue
                                );
                            } else {
                                gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::NZ, DataFormat::NZ>(
                                    l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 256],
                                    k_gm_tensor[kv_offset + block_size * 128 * embed_split_idx],
                                    qk_round_n,
                                    block_size,
                                    qk_round_n,
                                    256,
                                    256, 256);
                            }
                            SET_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                            WAIT_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                        } else if (embed_split_idx == 4) {
                            WAIT_FLAG(MTE1, MTE2, 2 + l1_kv_pingpong_flag);
                            now_l1_offset = l1_kv_pingpong_flag * 128 * 64 + 2 * 256 * 128;
                            if constexpr(KInputType == InputFormat::ND_FORMAT) {
                                gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::ND, DataFormat::NZ>(
                                    l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 64 + 2 * 256 * 128],
                                    k_rope_gm_tensor[kv_offset_rope],
                                    qk_n,         // nValue
                                    qk_round_n,             // dstNzC0Stride
                                    0,                     // dstNzMatrixStride, unused
                                    64,            // dValue
                                    0,                     // dstNzMatrixStride, unused
                                    stride_kv_rope            // srcDValue
                                );
                            } else {
                                gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::NZ, DataFormat::NZ>(
                                    l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 64 + 2 * 256 * 128],
                                    k_rope_gm_tensor[kv_offset_rope],
                                    qk_round_n,
                                    block_size,
                                    qk_round_n,
                                    64,
                                    64,
                                    64);
                            }
                            SET_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                            WAIT_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                        }
                        WAIT_FLAG(M, MTE1, embed_split_idx % 2 + 2);
                        l1_to_l0_b<ArchType::ASCEND_V220, IN_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                            l0b_buf_tensor[embed_split_idx % 2 * 16384],
                            l1kv_buf_addr_tensor[now_l1_offset + embed_split_idx % 2 * qk_round_n * 128],
                            0,
                            round_embed_split_size * qk_round_n / T_CUBE_MATRIX_SIZE,  // repeat
                            0,
                            1,                                        // srcStride
                            0,
                            0                                        // dstStride
                        );
                        if (embed_split_idx == 1 || embed_split_idx == 3) {
                            SET_FLAG(MTE1, MTE2, l1_kv_pingpong_flag);
                        }
                        if (embed_split_idx == 4) {
                            SET_FLAG(MTE1, MTE2, 2 + l1_kv_pingpong_flag);
                        }
                        SET_FLAG(MTE1, M, embed_split_idx % 2 + 2);
                        WAIT_FLAG(MTE1, M, embed_split_idx % 2);
                        WAIT_FLAG(MTE1, M, embed_split_idx % 2 + 2);
                        if (embed_split_idx == 0) {
                            WAIT_FLAG(FIX, M, l1_kv_pingpong_flag);
                        }
                        mmad<ArchType::ASCEND_V220, IN_DTYPE, IN_DTYPE, mm1OutputType, false>(
                            mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
                            l0a_buf_tensor[embed_split_idx % 2 * 16384],
                            l0b_buf_tensor[embed_split_idx % 2 * 16384],
                            m,     // m
                            qk_n,  // n
                            embed_split_size,   // k
                            embed_split_idx == 0     // cmatrixInitVal
                        );
                        PIPE_BARRIER(M);
                        SET_FLAG(M, MTE1, embed_split_idx % 2);
                        SET_FLAG(M, MTE1, embed_split_idx % 2 + 2);

                        // copy S to gm
                        if (embed_split_idx == 4) {
                            SET_FLAG(M, FIX, l1_kv_pingpong_flag);
                            WAIT_FLAG(M, FIX, l1_kv_pingpong_flag);

                            l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, mm1CopyType, mm1OutputType>(
                                s_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER * 4 + (uint64_t)((n_idx / s_block_stack) % 2) * TMP_SIZE_DECODER * 2 + split_idx * pp_n_scalar],
                                mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
                                m,           // MSize
                                qk_round_n,  // NSize
                                RoundUp<16>(m), // srcStride
                                sv_round_n  // dstStride_dst_D
                            );
                            SET_FLAG(FIX, M, l1_kv_pingpong_flag);
                        }
                    }
                }
                FftsCrossCoreSync<PIPE_FIX, 2>(QK_READY_DECODER);
            }
            /* ************ CUBE2 stage1  ************* */
            if (n_idx >= s_block_stack) {
                if (n_idx + s_block_stack > n_loop + s_block_stack - 1) {
                    sv_n = cur_kv_seqlen - (n_idx - s_block_stack) * pp_n_scalar; // delete
                } else {
                    sv_n = pp_n_scalar * s_block_stack;
                }
                uint32_t sv_round_n = (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                uint32_t embed_split_size = 128;
                embed_split_loop_v = 4;
                uint32_t round_embed_split_size = RoundUp<T_BLOCK_SIZE>(embed_split_size);
                // WaitFlagDev(SOFTMAX_READY_DECODER);
                for (uint32_t embed_split_idx = 0; embed_split_idx < embed_split_loop_v; ++embed_split_idx) {
                    uint32_t l0c_pingpong_flag = embed_split_idx % 2;
                    for (uint32_t split_idx = 0; split_idx < s_block_stack && n_idx + split_idx < n_loop + s_block_stack; split_idx++) {
                        uint32_t now_idx = n_idx + split_idx;
                        if (now_idx == (n_loop + s_block_stack - 1)) {
                            qk_n_2 = (cur_kv_seqlen - (now_idx - s_block_stack) * pp_n_scalar);
                        } else {
                            qk_n_2 = pp_n_scalar;
                        }
                        qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);
                        uint32_t l1_kv_pingpong_flag = now_idx % 2;
                        uint32_t l0_p_pingpong_flag = now_idx % 2;
                        uint32_t l0b_pingpong_flag = now_idx % 2;
                        uint32_t block_table_id = (uint32_t)(*(block_tables_gm +
                                        cur_batch * max_num_blocks_per_query + start_kv / block_size + now_idx - s_block_stack));
                        int64_t kv_offset = (int64_t)block_table_id * block_size * stride_kv + block_size * embed_split_idx * embed_split_size;
                        WAIT_FLAG(MTE1, MTE2, 4 + l1_kv_pingpong_flag);  // 等待V全部搬入L0B
                        if constexpr(KInputType == InputFormat::ND_FORMAT) {
                            gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::ND, DataFormat::NZ>(
                                l1kv_buf_addr_tensor[128 * 576 + 128 * 64 + l1_kv_pingpong_flag * 128 * 128],
                                k_gm_tensor[(int64_t)block_table_id * block_size * stride_kv + embed_split_idx * embed_split_size],
                                qk_n_2,         // nValue
                                qk_round_n_2,             // dstNzC0Stride
                                0,                     // dstNzMatrixStride, unused
                                128,            // dValue
                                0,                     // dstNzMatrixStride, unused
                                stride_kv            // srcDValue
                            );
                        } else {
                            gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::NZ, DataFormat::NZ>(
                                l1kv_buf_addr_tensor[128 * 576 + 128 * 64 + l1_kv_pingpong_flag * 128 * 128],
                                k_gm_tensor[kv_offset],
                                qk_round_n_2,
                                block_size,
                                qk_round_n_2,
                                128,
                                128, 128);
                        }
                        SET_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                        WAIT_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                        WAIT_FLAG(M, MTE1, l0b_pingpong_flag + 2);
                        AscendC::LoadData2dTransposeParams loadDataParams;
                        loadDataParams.dstGap = 0;
                        loadDataParams.startIndex = 0;
                        loadDataParams.dstFracGap = 0;
                        loadDataParams.repeatTimes = round_embed_split_size / T_BLOCK_SIZE;
                        loadDataParams.srcStride = qk_round_n_2 / T_BLOCK_SIZE;
                        uint16_t dstGap = sizeof(IN_DTYPE) == 1 ? 1 : 0;
                        loadDataParams.dstGap = dstGap;
                        for (uint32_t l0b_load_idx = 0; l0b_load_idx < qk_round_n_2 / T_BLOCK_SIZE; ++l0b_load_idx) {
                            // along embd dim
                            AscendC::LoadDataWithTranspose(
                                    l0b_buf_tensor[l0b_pingpong_flag * 16384 + l0b_load_idx * RoundUp<16>(embed_split_size) * T_BLOCK_SIZE],
                                    l1kv_buf_addr_tensor[128 * 576 + 128 * 64 + l1_kv_pingpong_flag * 128 * 128 + l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE],
                                    loadDataParams);
                        }

                        SET_FLAG(MTE1, MTE2, 4 + l1_kv_pingpong_flag);
                        // move p from gm to l1
                        uint32_t p_move_head_num = row_num;
                        if (embed_split_idx == 0 && split_idx == 0) {
                            WaitFlagDev(SOFTMAX_READY_DECODER);
                        }

                        WAIT_FLAG(MTE1, MTE2, l0_p_pingpong_flag + 6);
                        gm_to_l1<ArchType::ASCEND_V220, IN_DTYPE, DataFormat::ND, DataFormat::NZ>(
                            l1p_buf_addr_tensor[l0_p_pingpong_flag * 128 * 128],
                            p_gm_tensor[(uint64_t)block_idx * TMP_SIZE * 2 + ((n_idx / s_block_stack - 1) % 2) * TMP_SIZE + split_idx * 128],
                            p_move_head_num,         // nValue
                            (p_move_head_num + 15) / 16 * 16,// dstNzC0Stride
                            0,                     // dstNzMatrixStride, unused
                            qk_round_n_2,           // dValue
                            0,                     // dstNzMatrixStride, unused
                            sv_round_n           // srcDValue
                        );
                        SET_FLAG(MTE2, MTE1, EVENT_ID7);
                        WAIT_FLAG(MTE2, MTE1, EVENT_ID7);
                        // move p from l1 to l0a
                        WAIT_FLAG(M, MTE1, l0_p_pingpong_flag);
                        uint32_t p_load_coeff = RoundUp<16>(p_move_head_num);
                        for (uint64_t loa_load_idx = 0; loa_load_idx < p_load_coeff / BLOCK_SIZE; ++loa_load_idx) {
                            l1_to_l0_a<ArchType::ASCEND_V220, IN_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                                l0a_buf_tensor[l0_p_pingpong_flag * 16384 + loa_load_idx * qk_round_n_2 * BLOCK_SIZE],
                                l1p_buf_addr_tensor[l0_p_pingpong_flag * 128 * 128 + loa_load_idx * T_CUBE_MATRIX_SIZE],
                                0,
                                qk_round_n_2 / T_BLOCK_SIZE,                                 // repeat
                                0,
                                p_load_coeff / BLOCK_SIZE,                               // srcStride
                                0,
                                0                                                        // dstStride
                            );
                        }
                        SET_FLAG(MTE1, MTE2, l0_p_pingpong_flag + 6);

                        SET_FLAG(MTE1, M, l0b_pingpong_flag);
                        WAIT_FLAG(MTE1, M, l0b_pingpong_flag);
                        if (split_idx == 0) {
                            WAIT_FLAG(FIX, M, l0c_pingpong_flag);
                        }
                        mmad<ArchType::ASCEND_V220, IN_DTYPE, IN_DTYPE, mm2OutputType, false>(
                            mm2_l0c_buf_tensor[l0c_pingpong_flag * 16384],
                            l0a_buf_tensor[l0_p_pingpong_flag * 16384],
                            l0b_buf_tensor[l0b_pingpong_flag * 16384],
                            m,     // m
                            embed_split_size,   // n
                            qk_n_2,  // k
                            split_idx == 0     // cmatrixInitVal
                        );
                        PIPE_BARRIER(M);
                        SET_FLAG(M, MTE1, l0b_pingpong_flag + 2);
                        SET_FLAG(M, MTE1, l0_p_pingpong_flag);
                    }
                    SET_FLAG(M, FIX, l0c_pingpong_flag);
                    WAIT_FLAG(M, FIX, l0c_pingpong_flag);

                    // copy O to gm
                    l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, mm2CopyType, mm2OutputType>(
                        o_tmp_gm_tensor[(uint64_t)block_idx * TMP_SIZE * 2 + embed_split_idx * round_embed_split_size + ((n_idx / s_block_stack - 1) % 2) * TMP_SIZE],
                        mm2_l0c_buf_tensor[l0c_pingpong_flag * 16384],
                        m,        // MSize
                        RoundUp<16>(embed_split_size),  // NSize 32B align
                        RoundUp<16>(m),       // srcStride
                        round_v  // dstStride_dst_D
                    );
                    SET_FLAG(FIX, M, l0c_pingpong_flag);
                }
                FftsCrossCoreSync<PIPE_FIX, 2>(UPDATE_READY_DECODER);
            }
        }
    }

private:
    __gm__ IN_DTYPE *__restrict__ q_gm{nullptr};
    __gm__ IN_ROPE_DTYPE *__restrict__ q_rope_gm{nullptr};
    __gm__ IN_KVDTYPE *__restrict__ ctkv_gm{nullptr};
    __gm__ IN_KVDTYPE *__restrict__ k_gm{nullptr};
    __gm__ IN_ROPE_DTYPE *__restrict__ k_rope_gm{nullptr};
    __gm__ IN_KVDTYPE *__restrict__ v_gm{nullptr};

    __gm__ mm1CopyType *__restrict__ s_gm{nullptr};
    __gm__ float *__restrict__ s_rope_gm{nullptr};
    __gm__ IN_DTYPE *__restrict__ p_gm{nullptr};
    __gm__ mm2CopyType *__restrict__ o_tmp_gm{nullptr};
    __gm__ int32_t *__restrict__ block_tables_gm{nullptr};
    __gm__ uint8_t *__restrict__ tiling_gm{nullptr};

    AscendC::GlobalTensor<OUT_DTYPE> o_gm_tensor;
    AscendC::GlobalTensor<IN_DTYPE> q_gm_tensor;
    AscendC::GlobalTensor<IN_ROPE_DTYPE> q_rope_gm_tensor;
    AscendC::GlobalTensor<IN_KVDTYPE> k_gm_tensor;
    AscendC::GlobalTensor<IN_ROPE_DTYPE> k_rope_gm_tensor;
    AscendC::GlobalTensor<IN_KVDTYPE> v_gm_tensor;
    AscendC::GlobalTensor<mm1CopyType> s_gm_tensor;
    AscendC::GlobalTensor<float> s_rope_gm_tensor;
    AscendC::GlobalTensor<IN_DTYPE> p_gm_tensor;
    AscendC::GlobalTensor<mm2CopyType> o_tmp_gm_tensor;
    AscendC::GlobalTensor<int32_t> block_tables_gm_tensor;

    const uint32_t l1q_buf_addr_offset = 0;
    const uint32_t l1q_rope_buf_addr_offset = 65536;
    const uint32_t l1kv_buf_addr_offset = 147456;
    const uint32_t l1kv_rope_buf_addr_offset= 278528;
    const uint32_t l1p_buf_addr_offset = 442368;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<IN_DTYPE> l1q_buf_addr_tensor;
    AscendC::LocalTensor<IN_ROPE_DTYPE> l1q_rope_buf_addr_tensor;
    AscendC::LocalTensor<IN_DTYPE> l1kv_buf_addr_tensor;
    AscendC::LocalTensor<IN_ROPE_DTYPE> l1kv_rope_buf_addr_tensor;
    AscendC::LocalTensor<IN_DTYPE> l1p_buf_addr_tensor;

    AscendC::LocalTensor<IN_DTYPE> l0a_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, IN_DTYPE>(0);
    AscendC::LocalTensor<IN_DTYPE> l0b_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, IN_DTYPE>(0);
    AscendC::LocalTensor<mm1OutputType> mm1_l0c_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, mm1OutputType>(0);
    AscendC::LocalTensor<mm2OutputType> mm2_l0c_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, mm2OutputType>(0);


    uint32_t num_batches{0};
    uint32_t q_heads{0};
    uint32_t kv_heads{0};
    uint32_t embedding_size{0};
    uint32_t block_size{0};
    uint32_t max_num_blocks_per_query{0};
    uint32_t group_num{0};
    uint32_t stride_kv{0};
    uint32_t stride_kv_rope{0};
    uint32_t stride_vo{0};
    uint32_t m{0};
    uint32_t __k{0};
    uint32_t __v{0};
    uint32_t round_k{0};
    uint32_t round_v{0};
    uint32_t process_num{0};
    uint32_t tiling_head_size{0};
    uint32_t tiling_para_size{0};
    uint32_t block_size_calc{0};
    uint32_t mask_type{0};
    uint32_t totalTaskNum{0};
    uint32_t maxKVSeqLen{0};

    uint32_t cur_qn_blk_size{0};
    uint32_t num_batches_pad{0};

    uint32_t embed_split_size_qk{0};
    uint32_t embed_split_loop_qk{1};
    uint32_t embed_split_size_v{0};
    uint32_t embed_split_loop_v{1};

    uint32_t l1_pingpong_flag = 0;
    uint32_t l1b_pingpong_flag = 0;
    uint32_t l0_pingpong_flag = 0;
    uint32_t l0b_pingpong_flag = 0;
    uint32_t l0c_pingpong_flag = 0;
    uint32_t l1p_pingpong_flag = 0;

    uint32_t l1_offset = l1_pingpong_flag * L1_UINT8_BUF_SIZE_DECODER / sizeof(IN_DTYPE);
    uint32_t l1b_offset = l1b_pingpong_flag * L1_KV_UINT8_SIZE / sizeof(IN_DTYPE);
    uint32_t l1_scale_offset = l1_pingpong_flag * L1_SCALE_UINT64_SIZE;
    uint32_t l1_bias_offset = l1_pingpong_flag * L1_OFFSET_INT32_SIZE;
    uint32_t l0_offset = l0_pingpong_flag * L0AB_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
    uint32_t l0c_offset = l0c_pingpong_flag * L0C_FLOAT_BUF_SIZE;
    uint32_t l0b_offset = l0b_pingpong_flag * L0AB_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
    uint32_t l1p_start_offset = l1p_pingpong_flag * L1_P_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
};



#endif
#ifdef __DAV_C220_VEC__
template <TilingKeyType tilingKeyType = TilingKeyType::TILING_HALF_DATA, typename IN_DTYPE = half, typename OUT_DTYPE = half, bool IS_RING = false, BlockStack blockStack = BlockStack::ONE_FLOW, bool EnableOptimization = false>
class MLADecoderAiv{
public:
    using mm1OutputType = typename AttentionType<tilingKeyType>::mm1OutputType;
    using mm1CopyType = typename AttentionType<tilingKeyType>::mm1CopyType;
    using mmScaleType = typename AttentionType<tilingKeyType>::mmScaleType;
    using mm2OutputType = typename AttentionType<tilingKeyType>::mm2OutputType;
    using mm2CopyType = typename AttentionType<tilingKeyType>::mm2CopyType;
    static constexpr uint32_t T_BLOCK_SIZE =  BLOCK_SIZE_32 / sizeof(IN_DTYPE);
    static constexpr uint32_t T_BLOCK_OFFSET = 2 / sizeof(IN_DTYPE);

    __aicore__ __attribute__((always_inline)) inline MLADecoderAiv() {}

    __aicore__ __attribute__((always_inline)) inline void SetArgs(
        __gm__ uint8_t* __restrict__ gm_block_table,
        __gm__ uint8_t* __restrict__ deq_qk_in_gm,
        __gm__ uint8_t* __restrict__ deq_pv_in_gm,
        __gm__ uint8_t *__restrict__ o_out_gm,
        __gm__ uint8_t *__restrict__ s_out_gm,
        __gm__ uint8_t *__restrict__ s_rope_out_gm,
        __gm__ uint8_t *__restrict__ p_out_gm,
        __gm__ uint8_t *__restrict__ o_temp_gm,
        __gm__ uint8_t *__restrict__ globalo_gm,
        __gm__ uint8_t *__restrict__ tmp_gm,
        __gm__ uint8_t *__restrict__ tiling_para_gm,
        __gm__ uint8_t *__restrict__ mask_input_gm)
    {
        sub_block_idx = static_cast<uint64_t>(GetSubBlockidx());
        SetAtomicnone();
        SetMasknorm();
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        o_gm = reinterpret_cast<__gm__ OUT_DTYPE *>(o_out_gm);
        s_gm = reinterpret_cast<__gm__ mm1CopyType *>(s_out_gm);
        p_gm = reinterpret_cast<__gm__ IN_DTYPE *>(p_out_gm);
        o_tmp_gm = reinterpret_cast<__gm__ mm2CopyType *>(o_temp_gm);
        go_gm = reinterpret_cast<__gm__ float *>(globalo_gm);
        tiling_gm = reinterpret_cast<__gm__ uint8_t *>(tiling_para_gm);
        gm_block_tables_ = reinterpret_cast<__gm__ int32_t*>(gm_block_table);
        o_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_DTYPE *>(o_gm));
        mask_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_DTYPE *>(mask_input_gm));
        s_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ mm1CopyType *>(s_gm));
        p_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(p_gm));
        o_tmp_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ mm2CopyType *>(o_tmp_gm));
        go_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(go_gm));
        tmp_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tmp_gm));
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            deq_scale_gm_tensor_q1.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(deq_qk_in_gm));
            deq_scale_gm_tensor_k1.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(deq_pv_in_gm));
            s_rope_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(s_rope_out_gm));
        }

        num_batches = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm));
        q_heads = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_NUMHEADS));
        embedding_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADDIM));
        block_size = (int32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_BLOCKSIZE));
        max_num_blocks_per_query = (uint32_t)(*((__gm__ uint32_t*)tiling_para_gm + TILING_MAXBLOCKS));
        tor = (float)(*((__gm__ float *)tiling_para_gm + TILING_TOR));
        num_kv_heads = (uint32_t)(*((__gm__ uint32_t*)tiling_para_gm + TILING_KVHEADS));
        tiling_head_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADSIZE));
        tiling_para_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_PARASIZE));
        totalTaskNum = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + 13));
        maxKVSeqLen = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MAX_KV_SEQ_LEN));

        cur_qn_blk_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MTP_HEAD_SPLIT_SIZE));
        block_size_calc = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_BLOCKSIZE_CALC));
        mask_type = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MASK_TYPE_ND));

        go_flag_scalar = 1;
        gl_flag_scalar = 1;

        __k = embedding_size;
        round_k = RoundUp<T_BLOCK_SIZE>(__k);
        __v = embedding_size;
        round_v = RoundUp<BLOCK_SIZE>(__v);
    }

    __aicore__ __attribute__((always_inline)) inline void SetArgs2(
        __gm__ uint8_t *__restrict__ lse_out_gm)
    {
        lse_gm = reinterpret_cast<__gm__ OUT_DTYPE *>(lse_out_gm);
        lse_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_DTYPE *>(lse_gm));
    }

    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        SET_FLAG(MTE3, V, EVENT_ID0);
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        SET_FLAG(MTE3, MTE2, EVENT_ID2);
        SET_FLAG(MTE3, MTE2, EVENT_ID3);
        SET_FLAG(MTE3, MTE2, EVENT_ID4);
        SET_FLAG(V, MTE2, EVENT_ID4);
        SET_FLAG(V, MTE2, EVENT_ID0);
        SET_FLAG(MTE3, V, EVENT_ID2);
        SET_FLAG(V, MTE2, EVENT_ID2);


        uint64_t cur_batch = 0;

        uint32_t q_block_num_per_batch = (q_heads + cur_qn_blk_size - 1) / cur_qn_blk_size;
        uint32_t process_num = q_block_num_per_batch * num_batches;

        for (uint32_t process = block_idx; process < process_num; process += (uint32_t)block_num) {  // for task
            cur_batch = process / q_block_num_per_batch;
            if (cur_batch >= num_batches) break;

            uint32_t offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
            uint32_t start_core_idx = (cur_batch * q_block_num_per_batch) % block_num;

            uint32_t q_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));
            uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 1 + offset_tiling));
            if (kv_seqlen == 0) {
                continue;
            }
            uint32_t kv_seqlen_align = (kv_seqlen + block_size - 1) / block_size * block_size;

            uint32_t start_head = (process % q_block_num_per_batch) * cur_qn_blk_size;
            uint32_t start_kv = 0;
            uint32_t cur_q_seq_len = q_seqlen;
            uint32_t cur_kv_seqlen = kv_seqlen;
            uint32_t cur_head_num = cur_qn_blk_size;
            uint32_t cur_nIndx = 0;
            InnerRunVectorChange(cur_batch, start_head, cur_nIndx, cur_q_seq_len, cur_kv_seqlen, cur_head_num,
                offset_tiling, 512, embed_split_loop_v_former);
        }

        WAIT_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID3);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID4);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID4);
        WAIT_FLAG(MTE3, V, EVENT_ID2);
        WAIT_FLAG(V, MTE2, EVENT_ID2);
    }

    __aicore__ __attribute__((always_inline)) inline void RunTP1()
    {
        SET_FLAG(MTE3, V, EVENT_ID0);
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        SET_FLAG(MTE3, MTE2, EVENT_ID1);
        SET_FLAG(MTE3, MTE2, EVENT_ID2);
        SET_FLAG(MTE3, MTE2, EVENT_ID3);
        SET_FLAG(MTE3, MTE2, EVENT_ID4);
        SET_FLAG(V, MTE2, EVENT_ID4);
        SET_FLAG(V, MTE2, EVENT_ID0);
        SET_FLAG(MTE3, V, EVENT_ID2);
        SET_FLAG(V, MTE2, EVENT_ID2);

        uint32_t tail = totalTaskNum % block_num;
        if constexpr (EnableOptimization) {

        } else{
            tail = 0; // control whether to run tail optimization
        }
        uint32_t totalTaskNumRound = totalTaskNum - tail;
        
        for (uint32_t process = block_idx; process < totalTaskNumRound; process += (uint32_t)block_num) {  // for task
            uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
            uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));

            uint32_t q_seqlen = 1;
            uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
            if (kv_seqlen == 0) {
                continue;
            }
            uint32_t kv_seqlen_align = (kv_seqlen + block_size - 1) / block_size * block_size;

            uint32_t start_head = 0;
            uint32_t start_kv = 0;
            uint32_t cur_q_seq_len = q_seqlen;
            uint32_t cur_kv_seqlen = kv_seqlen;
            uint32_t cur_head_num = q_heads;
            uint32_t cur_nIndx = 0;

            InnerRunVectorChangeTP1(cur_batch, start_head, cur_nIndx, cur_q_seq_len, cur_kv_seqlen, cur_head_num, offset_tiling, 512, embed_split_loop_v_former);
        }
        
        // suppose all seqs have same length
        if (tail > 0){
            uint32_t sample_kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + tiling_head_size + 2));
            bool enableExtraOptimization = true;
            if (block_num % 4 == 3) {
                // cannot optimize this situation due to math problem
                enableExtraOptimization = false;
            }
            if (sample_kv_seqlen <= 2048){
                // Too Short to benefit from optimization
                enableExtraOptimization = false;
            }
            if (!enableExtraOptimization || tail <= block_num / 2) {
                // collect all metadata 
                uint32_t cores_per_seq = 1;
                // potential optimization space which not in the following branches, e.g. 1 seq with 24 cores, 2 seq with 12 cores, 3 seq with 8 cores
                if (0 < tail && tail <= block_num / 4) {// 6 tasks left, each works with 4 cores
                    cores_per_seq = 4;
                    if (tail == 1){
                        cores_per_seq = block_num;
                    }
                    else if (tail == 2){
                        cores_per_seq = block_num / 2;
                    }
                    else if(tail == 3){
                        cores_per_seq = block_num / 3;
                    }
                    else if(tail == 4){
                        cores_per_seq = block_num / 4;
                    }
                }
                else if(block_num / 4 < tail && tail <= block_num / 3) { // 8 tasks left, each works with 3 cores
                    cores_per_seq = 3;
                }
                else if(block_num / 3 < tail && tail <= block_num / 2) { // 12 tasks left, each works with 2 cores
                    cores_per_seq = 2;
                }
                else {
                    // no extra optimization for tail > 12
                    cores_per_seq = 1;
                }
                if (!enableExtraOptimization) {
                    cores_per_seq = 1;
                }
                uint32_t process = totalTaskNumRound + block_idx / cores_per_seq;
                if (process < totalTaskNum) {
                    uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
                    uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));
                    
                    uint32_t q_seqlen = 1;
                    uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
                    uint32_t kv_seqlen_each = kv_seqlen / cores_per_seq;
                    uint32_t kv_seqlen_align = (kv_seqlen_each + block_size - 1) / block_size * block_size;
                    uint32_t actual_work_cores = kv_seqlen / kv_seqlen_align + (kv_seqlen % kv_seqlen_align != 0);
                   
                    // cores_per_seq = actual_work_cores;
                    uint32_t kv_seqlen_process = 0;
                    if (block_idx < block_idx / cores_per_seq * cores_per_seq + actual_work_cores){
                        kv_seqlen_process = (block_idx % cores_per_seq == actual_work_cores - 1) ? 
                            (kv_seqlen - kv_seqlen_align * (actual_work_cores - 1)) : kv_seqlen_align;
                    }
                    uint32_t start_head = 0;
                    uint32_t start_kv = (block_idx % cores_per_seq) * kv_seqlen_align;
                    uint32_t cur_q_seq_len = q_seqlen;
                    uint32_t cur_kv_seqlen = kv_seqlen_process;
                    uint32_t cur_head_num = q_heads;
                    uint32_t cur_nIndx = 0; // no use, follow the previous code
        
                    if (cores_per_seq == 1 || actual_work_cores == 1){ // no optimization for tail
                        if (kv_seqlen > 0 && kv_seqlen_process > 0) {
                            InnerRunVectorChangeTP1(cur_batch, start_head, cur_nIndx, cur_q_seq_len, cur_kv_seqlen, cur_head_num, offset_tiling, 512, embed_split_loop_v_former);
                        }
                    }
                    else {
                        // customized InnerRunVectorChange for tail processing
                        if (kv_seqlen > 0 && kv_seqlen_process > 0) {
                            TailInnerRunVectorChangeTP1(start_head, cur_q_seq_len, cur_kv_seqlen, cur_head_num, offset_tiling, 512, embed_split_loop_v_former);
                        }
                        // Sync all vector cores 
                        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(TAIL_OPTIMIZATION_SYNC);
                        AscendC::CrossCoreWaitFlag<0x0, PIPE_MTE3>(TAIL_OPTIMIZATION_SYNC);
                        
                        // cores_per_seq >= 2
                        if (cores_per_seq >= 4){
                            // 8 vector cores
                            if (block_idx % cores_per_seq < 4) {
                                TailInnerGatherVectorTP1(start_head + cur_head_num / 4 * (block_idx % cores_per_seq), cur_q_seq_len, cur_head_num / 4, block_idx - (block_idx % cores_per_seq), actual_work_cores, offset_tiling);
                            }
                        }
                        else{
                            // 4 vector cores
                            if (block_idx % cores_per_seq < 2) {
                                TailInnerGatherVectorTP1(start_head + cur_head_num / 2 * (block_idx % cores_per_seq), cur_q_seq_len, cur_head_num / 2, block_idx - (block_idx % cores_per_seq), actual_work_cores, offset_tiling);
                            }
                        }
                    }

                }
                else{
                    if (cores_per_seq != 1){
                        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(TAIL_OPTIMIZATION_SYNC);
                        AscendC::CrossCoreWaitFlag<0x0, PIPE_MTE3>(TAIL_OPTIMIZATION_SYNC);
                    }
                }

            }
            else if (tail > 3 * block_num / 4){
                // no benefit for optimizing this situation
                uint32_t process = totalTaskNumRound + block_idx;
                if (process < totalTaskNum) {
                    uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
                    uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));
                    
                    uint32_t q_seqlen = 1;
                    uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
                    if (kv_seqlen > 0) {
                        uint32_t kv_seqlen_align = (kv_seqlen + block_size - 1) / block_size * block_size;
    
                        uint32_t start_head = 0;
                        uint32_t start_kv = 0;
                        uint32_t cur_q_seq_len = q_seqlen;
                        uint32_t cur_kv_seqlen = kv_seqlen;
                        uint32_t cur_head_num = q_heads;
                        uint32_t cur_nIndx = 0;
    
                        InnerRunVectorChangeTP1(cur_batch, start_head, cur_nIndx, cur_q_seq_len, cur_kv_seqlen, cur_head_num, offset_tiling, 512, embed_split_loop_v_former);
                    }
                }
            }
            else {
                // 18 >= tail >= 12 
                // first 12 tasks, two cores per task
                {
                    uint32_t cores_per_seq = 2;
                    uint32_t process = totalTaskNumRound + block_idx / cores_per_seq;
                    uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
                    uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));
                    
                    uint32_t q_seqlen = 1;
                    uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
                    uint32_t kv_seqlen_each = kv_seqlen / cores_per_seq;
                    uint32_t kv_seqlen_align = (kv_seqlen_each + block_size - 1) / block_size * block_size;
                    uint32_t kv_seqlen_process = (block_idx % cores_per_seq == cores_per_seq - 1) ? 
                    (kv_seqlen - kv_seqlen_align * (cores_per_seq - 1)) : kv_seqlen_align;
                    
                    if (kv_seqlen > 0 && kv_seqlen_process > 0) {
                        uint32_t start_head = 0;
                        uint32_t start_kv = (block_idx % cores_per_seq) * kv_seqlen_align;
                        uint32_t cur_q_seq_len = q_seqlen;
                        uint32_t cur_kv_seqlen = kv_seqlen_process;
                        uint32_t cur_head_num = q_heads;
                        uint32_t cur_nIndx = 0; // no use, follow the previous code
            
                        // customized InnerRunVectorChange for tail processing
                        TailInnerRunVectorChangeTP1(start_head, cur_q_seq_len, cur_kv_seqlen, cur_head_num, offset_tiling, 512, embed_split_loop_v_former);
                        // Sync all vector cores 
                        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(TAIL_OPTIMIZATION_SYNC);
                        AscendC::CrossCoreWaitFlag<0x0, PIPE_MTE3>(TAIL_OPTIMIZATION_SYNC);
                        // 4 vector cores
                        if (block_idx % cores_per_seq < 2) {
                            TailInnerGatherVectorTP1(start_head + cur_head_num / 2 * (block_idx % cores_per_seq), cur_q_seq_len, cur_head_num / 2, block_idx - (block_idx % cores_per_seq), cores_per_seq, offset_tiling);
                        }
                    }
                    else{
                        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(TAIL_OPTIMIZATION_SYNC);
                        AscendC::CrossCoreWaitFlag<0x0, PIPE_MTE3>(TAIL_OPTIMIZATION_SYNC);
                    }
                }

                // other tasks
                {
                    uint32_t cores_per_seq = 4;
                    // TODO: check whether need extra sync between aic and aiv for used s_gm
                    uint32_t process = totalTaskNumRound + block_num / 2 + block_idx / cores_per_seq;
                    if (process < totalTaskNum) {
                        uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
                        uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));
                        
                        uint32_t q_seqlen = 1;
                        uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
                        uint32_t kv_seqlen_each = kv_seqlen / cores_per_seq;
                        uint32_t kv_seqlen_align = (kv_seqlen_each + block_size - 1) / block_size * block_size;
                        uint32_t kv_seqlen_process = (block_idx % cores_per_seq == cores_per_seq - 1) ? 
                        (kv_seqlen - kv_seqlen_align * (cores_per_seq - 1)) : kv_seqlen_align;
                        
                        if (kv_seqlen > 0 && kv_seqlen_process > 0) {
                            uint32_t start_head = 0;
                            uint32_t start_kv = (block_idx % cores_per_seq) * kv_seqlen_align;
                            uint32_t cur_q_seq_len = q_seqlen;
                            uint32_t cur_kv_seqlen = kv_seqlen_process;
                            uint32_t cur_head_num = q_heads;
                            uint32_t cur_nIndx = 0; // no use, follow the previous code
                            // customized InnerRunVectorChange for tail processing
                            TailInnerRunVectorChangeTP1(start_head, cur_q_seq_len, cur_kv_seqlen, cur_head_num, offset_tiling, 512, embed_split_loop_v_former);
                            // Sync all vector cores 
                            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(TAIL_OPTIMIZATION_SYNC);
                            AscendC::CrossCoreWaitFlag<0x0, PIPE_MTE3>(TAIL_OPTIMIZATION_SYNC);
                            
                            // 8 vector cores
                            if (block_idx % cores_per_seq < 4) {
                                TailInnerGatherVectorTP1(start_head + cur_head_num / 4 * (block_idx % cores_per_seq), cur_q_seq_len, cur_head_num / 4, block_idx - (block_idx % cores_per_seq), cores_per_seq, offset_tiling);
                            }
                            
                        }
                        else{
                            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(TAIL_OPTIMIZATION_SYNC);
                            AscendC::CrossCoreWaitFlag<0x0, PIPE_MTE3>(TAIL_OPTIMIZATION_SYNC);
                        }
                    }
                    else{
                        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(TAIL_OPTIMIZATION_SYNC);
                        AscendC::CrossCoreWaitFlag<0x0, PIPE_MTE3>(TAIL_OPTIMIZATION_SYNC);
                    }
                }
            }
        }

        WAIT_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID3);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID4);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID4);
        WAIT_FLAG(MTE3, V, EVENT_ID2);
        WAIT_FLAG(V, MTE2, EVENT_ID2);
    }
private:


   __aicore__ __attribute__((always_inline)) inline void ReduceMaxRepeatM(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        const AscendC::LocalTensor<float>& tempTensor,
        uint32_t sub_m,
        uint32_t qk_n,
        uint32_t qk_round_n)
    {
        if (qk_n <= FLOAT_VECTOR_SIZE) {
            __set_mask(qk_n);
            cmax_v<ArchType::ASCEND_V220, float, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(dst,
                src,
                sub_m,                    // repeat
                1,                        // dstRepeatStride
                1,                        // srcBlockStride
                qk_round_n / FLOAT_BLOCK_SIZE   // srcRepeatStride
            );
        } else {
            ub_to_ub<ArchType::ASCEND_V220, float>(
                tempTensor,
                src,
                0,                                             // sid
                sub_m,                                         // nBurst
                HALF_VECTOR_SIZE / BLOCK_SIZE,                 // lenBurst
                (qk_round_n - FLOAT_VECTOR_SIZE) / FLOAT_BLOCK_SIZE,  // srcGap
                0                                              // dstGap
            );
            PIPE_BARRIER(V);
            for (uint32_t rowmax_idx = 1; rowmax_idx < qk_n / FLOAT_VECTOR_SIZE; ++rowmax_idx) {
                max_v<ArchType::ASCEND_V220, float>(
                    tempTensor,
                    tempTensor,
                    src[rowmax_idx * FLOAT_VECTOR_SIZE],
                    sub_m,                         // repeat
                    1,                             // dstBlockStride
                    1,                             // src0BlockStride
                    1,                             // src1BlockStride
                    8,                             // dstRepeatStride
                    8,                             // src0RepeatStride
                    qk_round_n / FLOAT_BLOCK_SIZE  // src1RepeatStride
                );
            PIPE_BARRIER(V);
            }
            if (qk_n % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(qk_n % FLOAT_VECTOR_SIZE);
                max_v<ArchType::ASCEND_V220, float>(
                    tempTensor,
                    tempTensor,
                    src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    sub_m,                         // repeat
                    1,                             // dstBlockStride
                    1,                             // src0BlockStride
                    1,                             // src1BlockStride
                    8,                             // dstRepeatStride
                    8,                             // src0RepeatStride
                    qk_round_n / FLOAT_BLOCK_SIZE  // src1RepeatStride
                );
            }
            PIPE_BARRIER(V);
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            cmax_v<ArchType::ASCEND_V220, float, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
                dst,
                tempTensor,
                sub_m,      // repeat
                1,          // dstRepeatStride
                1,          // srcBlockStride
                8           // srcRepeatStride
            );
        }
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        PIPE_BARRIER(V);
    }


    __aicore__ __attribute__((always_inline)) inline void ReduceSumRepeatM(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        uint32_t sub_m,
        uint32_t qk_n,
        uint32_t qk_round_n)
    {
        if (qk_n <= FLOAT_VECTOR_SIZE) {
            __set_mask(qk_n);
            cadd_v<ArchType::ASCEND_V220, float>(
                dst,
                src,
                sub_m,           // repeat
                1,               // dstRepeatStride
                1,               // srcBlockStride
                qk_round_n / FLOAT_BLOCK_SIZE   // srcRepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else {
            for (uint32_t rowsum_idx = 1; rowsum_idx < qk_n / FLOAT_VECTOR_SIZE; ++rowsum_idx) {
                add_v<ArchType::ASCEND_V220, float>(
                    src,
                    src,
                    src[rowsum_idx * FLOAT_VECTOR_SIZE],
                    sub_m,           // repeat
                    1,               // dstBlockStride
                    1,               // src0BlockStride
                    1,               // src1BlockStride
                    qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    qk_round_n / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    qk_round_n / FLOAT_BLOCK_SIZE   // src1RepeatStride
                );
                PIPE_BARRIER(V);
            }
            if (qk_n % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(qk_n % FLOAT_VECTOR_SIZE);
                add_v<ArchType::ASCEND_V220, float>(
                    src,
                    src,
                    src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    sub_m,           // repeat
                    1,               // dstBlockStride
                    1,               // src0BlockStride
                    1,               // src1BlockStride
                    qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    qk_round_n / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    qk_round_n / FLOAT_BLOCK_SIZE   // src1RepeatStride
                );
                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            PIPE_BARRIER(V);

            cadd_v<ArchType::ASCEND_V220, float>(
                dst,
                src,
                sub_m,           // repeat
                1,               // dstRepeatStride
                1,               // srcBlockStride
                qk_round_n / FLOAT_BLOCK_SIZE   // srcRepeatStride
            );
        }
    }

    __aicore__ __attribute__((always_inline)) inline void TensorSubValueRepeatM(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        const AscendC::LocalTensor<float>& MaxTensor,
        const AscendC::LocalTensor<float>& tempMaxTensor,
        uint32_t sub_m,
        uint32_t round_sub_m,
        uint32_t qk_n,
        uint32_t qk_round_n)
    {
        brcb_v<ArchType::ASCEND_V220, uint32_t>(
            tempMaxTensor.ReinterpretCast<uint32_t>(),
            MaxTensor.ReinterpretCast<uint32_t>(),
            1,               // dstBlockStride
            8,               // dstRepeatStride
            round_sub_m / FLOAT_BLOCK_SIZE  // repeat
        );
        PIPE_BARRIER(V);
        for (uint32_t sub_v_idx = 0; sub_v_idx < qk_n / FLOAT_VECTOR_SIZE; ++sub_v_idx) {
            sub_v<ArchType::ASCEND_V220, float>(dst[sub_v_idx * FLOAT_VECTOR_SIZE],
                src[sub_v_idx * FLOAT_VECTOR_SIZE],
                tempMaxTensor,
                sub_m,                    // repeat
                1,                        // dstBlockStride
                1,                        // src0BlockStride
                0,                        // src1BlockStride
                qk_round_n / FLOAT_BLOCK_SIZE, // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE, // src0RepeatStride
                1                         // src1RepeatStride
            );
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            sub_v<ArchType::ASCEND_V220, float>(dst[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tempMaxTensor,
                sub_m,                    // repeat
                1,                        // dstBlockStride
                1,                        // src0BlockStride
                0,                        // src1BlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1                         // src1RepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);
    }

    __aicore__ __attribute__((always_inline)) inline void TensorDivRepeatM(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        const AscendC::LocalTensor<float>& src1,
        uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n)
    {
        PIPE_BARRIER(V);
        for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE; ++vadd_idx) {
            div_v<ArchType::ASCEND_V220, float>(dst[vadd_idx * FLOAT_VECTOR_SIZE],
                src[vadd_idx * FLOAT_VECTOR_SIZE],
                src1,
                sub_m,                                  // repeat
                1,                                      // dstBlockStride
                1,                                      // src0BlockStride
                0,                                     // src1BlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,          // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE,          // src0RepeatStride
                1                                       // src1RepeatStride
            );
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            div_v<ArchType::ASCEND_V220, float>(dst[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                src1,
                sub_m,                                   // repeat
                1,                                      // dstBlockStride
                1,                                      // src0BlockStride
                0,                        // src1BlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,          // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE,         // src0RepeatStride
                1                                      // src1RepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);
    }

    __aicore__ __attribute__((always_inline)) inline void TensorMulRepeatM(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        const AscendC::LocalTensor<float>& src1,
        uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n, uint32_t src1BlockStride
    ) {
        PIPE_BARRIER(V);
        for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE; ++vadd_idx) {
            mul_v<ArchType::ASCEND_V220, float>(dst[vadd_idx * FLOAT_VECTOR_SIZE],
                src[vadd_idx * FLOAT_VECTOR_SIZE],
                src1,
                sub_m,                                  // repeat
                1,                                      // dstBlockStride
                1,                                      // src0BlockStride
                src1BlockStride,                        // src1BlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,          // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE,          // src0RepeatStride
                1                                       // src1RepeatStride
            );
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            mul_v<ArchType::ASCEND_V220, float>(dst[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                src1,
                sub_m,                                   // repeat
                1,                                      // dstBlockStride
                1,                                      // src0BlockStride
                src1BlockStride,                        // src1BlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,          // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE,         // src0RepeatStride
                1                                      // src1RepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);
    }

    __aicore__ __attribute__((always_inline)) inline void DeQuantPerHeadImpl(
        const AscendC::GlobalTensor<mmScaleType>& deScaleGm,
        const AscendC::GlobalTensor<int32_t>& src,
        AscendC::LocalTensor<float> dst,
        AscendC::LocalTensor<int32_t> temp,
        AscendC::LocalTensor<mmScaleType> deScaleUb,
        AscendC::LocalTensor<mmScaleType> tempScale,
        AscendC::LocalTensor<float> quantScale,
        uint32_t sub_m,
        uint32_t qk_n,
        uint32_t qk_round_n,
        bool online,
        bool move_tensor
    ){
        gm_to_ub_align<ArchType::ASCEND_V220, mmScaleType>(deScaleUb,
                                                        deScaleGm,
                                                        0,                                      // sid
                                                        1,                                      // nBurst
                                                        sub_m * sizeof(mmScaleType),             // lenBurst
                                                        0,                                      // leftPaddingNum
                                                        0,                                      // rightPaddingNum
                                                        0,                                      // srcGap
                                                        0                                       // dstGap
        );
        if (online) {
            // if dequant online need mul p quant scale
            SET_FLAG(MTE2, V, EVENT_ID2);
            WAIT_FLAG(MTE2, V, EVENT_ID2);
            TensorMulRepeatM(deScaleUb, deScaleUb, quantScale, 1, sub_m, RoundUp<16>(sub_m), 1);
        }

        if (move_tensor) {
            gm_to_ub<ArchType::ASCEND_V220, int32_t>(
                temp,
                src,
                0,                        // sid
                1,                        // nBurst
                CeilDiv<FLOAT_BLOCK_SIZE>(sub_m * qk_round_n),  // lenBurst
                0,                        // srcGap
                0                         // dstGap
            );
        }
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        brcb_v<ArchType::ASCEND_V220, uint32_t>(
            tempScale.template ReinterpretCast<uint32_t>(),
            deScaleUb.template ReinterpretCast<uint32_t>(),
            1,               // dstBlockStrides
            8,               // dstRepeatStride
            RoundUp<16>(sub_m) / FLOAT_BLOCK_SIZE  // repeat
        );
        PIPE_BARRIER(V);
        uint32_t count = sub_m * qk_round_n;
        uint32_t repeat_times = (count + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE;
        if (repeat_times < 255) {
            conv_v<ArchType::ASCEND_V220, int32_t, float>(
                dst, // dst
                temp, // src
                repeat_times,                  // repeat_times
                1,                            // dstBlockStride
                1,                            // srcBlockStride
                8,                            // dstRepeatStride
                8                             // srcRepeatStride
            );
        } else {
            for (uint64_t vconv_idx = 0; vconv_idx < 2; ++vconv_idx) {
                conv_v<ArchType::ASCEND_V220, int32_t, float>(
                    dst[vconv_idx * count / 2], // dst
                    temp[vconv_idx * count / 2], // src
                    (count / 2 + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,             // repeat_times
                    1,                                                                   // dstBlockStride
                    1,                                                                   // srcBlockStride
                    8,                                                                   // dstRepeatStride
                    8                                                                    // srcRepeatStride
                );
            }
        }
        TensorMulRepeatM(dst, dst, tempScale, sub_m, qk_n, qk_round_n, 0);
        PIPE_BARRIER(V);
    }

    __aicore__ __attribute__((always_inline)) inline void QuantPerTokenImpl(
        const AscendC::LocalTensor<IN_DTYPE>& dst,
        const AscendC::LocalTensor<float>& src,
        const AscendC::LocalTensor<float>& scale,
        uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n, uint32_t pQuantOnline)
    {
        if (pQuantOnline) {
            // scr / scale 
            TensorDivRepeatM(dst.template ReinterpretCast<float>(), src, scale, sub_m, qk_n, qk_round_n);
        } else {
            // scr * scale
            TensorMulRepeatM(dst.template ReinterpretCast<float>(), src, scale, sub_m, qk_n, qk_round_n, 0);
        }
        // src fp32 -> casttofp16 -> casttoint8
        uint32_t count = sub_m * qk_round_n;
        uint32_t repeat_times = (count + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE;
        if (repeat_times < 255) {
            conv_v<ArchType::ASCEND_V220, float, half>(
                dst.template ReinterpretCast<half>(), // dst
                dst.template ReinterpretCast<float>(), // src
                repeat_times,                  // repeat_times
                1,                            // dstBlockStride
                1,                            // srcBlockStride
                4,                            // dstRepeatStride
                8                             // srcRepeatStride
            );
        } else {
            for (uint64_t vconv_idx = 0; vconv_idx < 2; ++vconv_idx) {
                conv_v<ArchType::ASCEND_V220, float, half>(
                    dst.template ReinterpretCast<half>()[vconv_idx * count / 2], // dst
                    dst.template ReinterpretCast<float>()[vconv_idx * count / 2], // src
                    (count / 2 + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,             // repeat_times
                    1,                                                                   // dstBlockStride
                    1,                                                                   // srcBlockStride
                    4,                                                                   // dstRepeatStride
                    8                                                                    // srcRepeatStride
                );
            }
        }
        PIPE_BARRIER(V);
        for (uint32_t row_idx = 0; row_idx < qk_n / HALF_VECTOR_SIZE; ++row_idx) {
            AscendC::Cast<int8_t, half, false>(dst.template ReinterpretCast<int8_t>()[row_idx * HALF_VECTOR_SIZE],
                                               dst.template ReinterpretCast<half>()[row_idx * HALF_VECTOR_SIZE], AscendC::RoundMode::CAST_RINT,
                                               (uint64_t)0, sub_m, {1, 1, (uint8_t)((qk_round_n) / BLOCK_SIZE), (uint8_t)(qk_round_n / BLOCK_SIZE)});
        }
        if (qk_n % HALF_VECTOR_SIZE > 0) {
            __set_mask(qk_n % HALF_VECTOR_SIZE);
            AscendC::Cast<int8_t, half, false>(dst.template ReinterpretCast<int8_t>()[qk_n / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                                               dst.template ReinterpretCast<half>()[qk_n / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE], AscendC::RoundMode::CAST_RINT,
                                               (uint64_t)0, sub_m, {1, 1, (uint8_t)((qk_round_n) / BLOCK_SIZE), (uint8_t)(qk_round_n / BLOCK_SIZE)});
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);
    }

   __aicore__ __attribute__((always_inline)) inline void SoftmaxStage1(
        AscendC::GlobalTensor<IN_DTYPE> p_gm_tensor,
        AscendC::GlobalTensor<mm1CopyType> s_gm_tensor,
        AscendC::GlobalTensor<float> s_rope_gm_tensor,
        AscendC::GlobalTensor<OUT_DTYPE> mask_gm_tensor,
        AscendC::LocalTensor<float> dm32_ubuf_tensor,
        AscendC::LocalTensor<float> ll_ubuf_tensor,
        AscendC::LocalTensor<float> pm32_ubuf_tensor,
        uint32_t n_idx,
        uint32_t qk_n,
        uint32_t qk_round_n,
        uint32_t sub_m,
        uint32_t mask_offset,
        const uint32_t sub_n_loop,
        const uint32_t cur_batch,
        const uint32_t start_kv,
        const uint32_t real_n_loop,
	    const uint32_t head_idx,
        const uint32_t pm_flag_scalar,
        uint32_t cur_q_seqlen,
        uint32_t cur_kv_seqlen,
        bool need_mask
    )
    {
        uint32_t sub_m_d128 = (sub_m + 127) / 128;  // up aligned to 128
        uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 128
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;
        float quantMax = (float)1 / (float)127;
        WAIT_FLAG(V, MTE2, EVENT_ID2);
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            DeQuantPerHeadImpl(
                deq_scale_gm_tensor_q1[head_idx],
                s_gm_tensor,
                ls32_quant_ubuf_tensor, ls32_quant_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
                descale_q1_ubuf_tensor, tv32_ubuf_tensor, pm32_ubuf_tensor, sub_m, qk_n, qk_round_n, 0, 1);
            gm_to_ub<ArchType::ASCEND_V220, float>(
                ls32_ubuf_tensor.template ReinterpretCast<float>(),
                s_rope_gm_tensor,
                0,                        // sid
                1,                        // nBurst
                sub_m * qk_round_n / FLOAT_BLOCK_SIZE,
                0,                        // srcGap
                0                         // dstGap
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
            AscendC::Add(ls32_ubuf_tensor, ls32_ubuf_tensor, ls32_quant_ubuf_tensor, sub_m * qk_round_n); // float
            PIPE_BARRIER(V);
        } else {
            gm_to_ub<ArchType::ASCEND_V220, mm1CopyType>(
                ls32_ubuf_tensor.template ReinterpretCast<mm1CopyType>(),
                s_gm_tensor,
                0,                        // sid
                1,                        // nBurst
                sub_m * qk_round_n / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                        // srcGap
                0                         // dstGap
            );

            // TODO add mask type condition
            if (mask_type == 3) {
                uint32_t aligned_mask_copy_len = RoundUp<BLOCK_SIZE>(qk_n); // 16
                uint32_t mask_dst_stride = (qk_round_n -  aligned_mask_copy_len) / BLOCK_SIZE; // 0

                AscendC::DataCopyPad(
                    mask_ubuf_tensor,
                    mask_gm_tensor,
                    AscendC::DataCopyExtParams(
                        cur_q_seqlen,
                        qk_n * 2,
                        maxKVSeqLen * 2 - qk_n * 2,
                        mask_dst_stride,
                    0),
                    AscendC::DataCopyPadExtParams<OUT_DTYPE>(false, 0, 0, 0)
                );
            } else if (need_mask && mask_type == 4) {
                AscendC::DataCopy(
                    mask_ubuf_tensor,
                    mask_gm_tensor,
                    AscendC::DataCopyParams(
                        cur_q_seqlen,   // blockCount
                        qk_round_n * 2 / 32, // blockLen, 2 is sizeof(half)
                        MASK_COLUMNS * 2 / 32 - qk_round_n * 2 / 32, // srcStride
                        0 // dstStride
                        )
                );
            }

            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);

            if (mask_type == 3 || (need_mask && mask_type == 4)) {
                AscendC::Cast(
                    mask32_ubuf_tensor,
                    mask_ubuf_tensor,
                    AscendC::RoundMode::CAST_NONE,
                    cur_q_seqlen * qk_round_n);
            }
        }

        for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE; ++vadd_idx) {
            muls_v<ArchType::ASCEND_V220, float>(ls32_ubuf_tensor[vadd_idx * FLOAT_VECTOR_SIZE],
                ls32_ubuf_tensor[vadd_idx * FLOAT_VECTOR_SIZE],
                tor,
                sub_m,                          // repeat
                1,                              // dstBlockStride
                1,                              // srcBlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE  // srcRepeatStride
            );
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            muls_v<ArchType::ASCEND_V220, float>(ls32_ubuf_tensor[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                ls32_ubuf_tensor[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tor,
                sub_m,                          // repeat
                1,                              // dstBlockStride
                1,                              // srcBlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE  // srcRepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);

        if constexpr (tilingKeyType != TilingKeyType::TILING_INT8_DATA) {
            if (mask_type == 3 || (need_mask && mask_type == 4)) {
                uint32_t cur_compute_head_num = sub_m / cur_q_seqlen;
                for (uint32_t i = 0; i < cur_compute_head_num; i++) {
                    Add(
                        ls32_ubuf_tensor[cur_q_seqlen * qk_round_n * i],
                        ls32_ubuf_tensor[cur_q_seqlen * qk_round_n * i],
                        mask32_ubuf_tensor,
                        cur_q_seqlen * qk_round_n
                    );
                }
                PIPE_BARRIER(V);
            }
        }

        // *** lm = rowmax(ls)
        ReduceMaxRepeatM(lm32_ubuf_tensor, ls32_ubuf_tensor, lp32_ubuf_tensor, sub_m, qk_n, qk_round_n);
        // ReduceMaxChange(lm32_ubuf_tensor, ls32_ubuf_tensor, tv32_ubuf_tensor, round_sub_m, qk_n, qk_round_n);
        if (n_idx != 0) {
            // *** hm = vmax(lm, gm)
            max_v<ArchType::ASCEND_V220, float>(hm32_ubuf_tensor,
                lm32_ubuf_tensor,
                gm32_ubuf_tensor,
                sub_m_d64,  // repeat
                1,           // dstBlockStride
                1,           // src0BlockStride
                1,           // src1BlockStride
                8,           // dstRepeatStride
                8,           // src0RepeatStride
                8            // src1RepeatStride
            );
            PIPE_BARRIER(V);
            // *** dm = gm - hm
            sub_v<ArchType::ASCEND_V220, float>(dm32_ubuf_tensor,
                gm32_ubuf_tensor,
                hm32_ubuf_tensor,
                sub_m_d64,  // repeat
                1,           // dstBlockStride
                1,           // src0BlockStride
                1,           // src1BlockStride
                8,           // dstRepeatStride
                8,           // src0RepeatStride
                8            // src1RepeatStride
            );
            PIPE_BARRIER(V);
        } else {
            // *** hm = lm
            ub_to_ub<ArchType::ASCEND_V220, float>(
                hm32_ubuf_tensor,
                lm32_ubuf_tensor,
                0,                         // sid
                1,                         // nBurst
                round_sub_m / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                         // srcGap
                0                          // dstGap
            );
            PIPE_BARRIER(V);
        }
        // *** gm = hm
        ub_to_ub<ArchType::ASCEND_V220, float>(
            gm32_ubuf_tensor,
            hm32_ubuf_tensor,
            0,                         // sid
            1,                         // nBurst
            round_sub_m / FLOAT_BLOCK_SIZE,  // lenBurst
            0,                         // srcGap
            0                          // dstGap
        );
        PIPE_BARRIER(V);
        // *** hm_block = expand_to_block(hm)

        // *** ls = ls - hm_block
        TensorSubValueRepeatM(ls32_ubuf_tensor, ls32_ubuf_tensor,
                           hm32_ubuf_tensor, tv32_ubuf_tensor,
                           sub_m, round_sub_m, qk_n, qk_round_n);
        // *** ls = exp(ls)
        exp_v<ArchType::ASCEND_V220, float>(ls32_ubuf_tensor,
            ls32_ubuf_tensor,
            (sub_m * qk_round_n + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
            1,                               // dstBlockStride
            1,                               // srcBlockStride
            8,                               // dstRepeatStride
            8                                // srcRepeatStride
        );
        PIPE_BARRIER(V);
        // *** lp = castfp32to16(ls)
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            sub_v<ArchType::ASCEND_V220, float>(pm32_ubuf_tensor,
                lm32_ubuf_tensor,
                hm32_ubuf_tensor,
                sub_m_d64,   // repeat
                1,           // dstBlockStride
                1,           // src0BlockStride
                1,           // src1BlockStride
                8,           // dstRepeatStride
                8,           // src0RepeatStride
                8            // src1RepeatStride
            );
            PIPE_BARRIER(V);
            exp_v<ArchType::ASCEND_V220, float>(pm32_ubuf_tensor,
                pm32_ubuf_tensor,
                sub_m_d64,  // repeat
                1,                               // dstBlockStride
                1,                               // srcBlockStride
                8,                               // dstRepeatStride
                8                                // srcRepeatStride
            );
            PIPE_BARRIER(V);
            muls_v<ArchType::ASCEND_V220, float>(pm32_ubuf_tensor,
                pm32_ubuf_tensor,
                quantMax,
                sub_m_d64,              // repeat
                1,                      // dstBlockStride
                1,                      // srcBlockStride
                8,                      // dstRepeatStride
                8                        // srcRepeatStride
            );
            PIPE_BARRIER(V);
            brcb_v<ArchType::ASCEND_V220, uint32_t>(
                tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                pm32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                1,               // dstBlockStride
                8,               // dstRepeatStride
                round_sub_m / FLOAT_BLOCK_SIZE  // repeat
            );
            QuantPerTokenImpl(lp_ubuf_tensor, ls32_ubuf_tensor, tv32_ubuf_tensor, sub_m, qk_n, qk_round_n, 1);
        } else {
            conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(lp_ubuf_tensor,
                ls32_ubuf_tensor,
                (sub_m * qk_round_n + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
                1,                               // dstBlockStride
                1,                               // srcBlockStride
                4,                               // dstRepeatStride
                8                                // srcRepeatStride
            );
            PIPE_BARRIER(V);
        }
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        ub_to_gm<ArchType::ASCEND_V220, IN_DTYPE>(
            p_gm_tensor,
            lp_ubuf_tensor,
            0,                        // sid
            1,                        // nBurst
            sub_m * qk_round_n * T_BLOCK_OFFSET / T_BLOCK_SIZE,  // lenBurst
            0,                        // srcGap
            0                         // dstGap
        );

        // *** ll = rowsum(ls32)
        ReduceSumRepeatM(ll_ubuf_tensor, ls32_ubuf_tensor, sub_m, qk_n, qk_round_n);
        SET_FLAG(V, MTE2, EVENT_ID2);
        PIPE_BARRIER(V);
    }

    __aicore__ __attribute__((always_inline)) inline void SoftmaxStage2MLAHeadLoop(
        AscendC::GlobalTensor<mm2CopyType> o_tmp_gm_tensor,
        AscendC::GlobalTensor<float> go_gm_tensor,
        AscendC::GlobalTensor<OUT_DTYPE> o_gm_tensor,
        AscendC::LocalTensor<float> dm32_ubuf_tensor,
        AscendC::LocalTensor<float> ll_ubuf_tensor,
        AscendC::LocalTensor<float> pm32_ubuf_tensor,
        uint32_t n_idx,
        uint32_t n_loop,
        uint32_t qk_n,
        uint32_t qk_round_n,
        uint32_t sub_m,
        uint64_t o_offset,
        uint32_t head_idx,
        uint32_t pm_flag_scalar,
        uint32_t head_loop,
        uint32_t head_loop_idx,
        uint32_t q_seq_len,
        uint32_t sub_head_num,
        uint32_t cur_head_num,
        uint32_t numhead_per_process,
        uint32_t head_res_row_num,
        uint32_t head_start_sblock_idx,
        uint32_t tail_res_row_num
        )
    {
        uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 64
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        if (n_idx != 0) {
            gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
                lo_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
                o_tmp_gm_tensor,
                0,                    // sid
                1,                    // nBurst
                sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                    // srcGap
                0                     // dstGap
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
            if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
               DeQuantPerHeadImpl(
                    deq_scale_gm_tensor_k1[head_idx],
                    o_tmp_gm_tensor,
                    lo_ubuf_tensor, lo_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),// lo_ubuf_tensor use the same ptr
                    descale_k1_ubuf_tensor, tv32_ubuf_tensor, pm32_ubuf_tensor, sub_m, round_v, round_v, 1, 0);
            }
        }
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID4);
        if (n_idx != 0) {
            // *** dm = exp(dm)
            if (head_loop_idx == 0) {
                exp_v<ArchType::ASCEND_V220, float>(dm32_ubuf_tensor,
                    dm32_ubuf_tensor,
                    sub_m_d64,  // repeat
                    1,          // dstBlockStride
                    1,          // srcBlockStride
                    8,          // dstRepeatStride
                    8           // srcRepeatStride
                );
                PIPE_BARRIER(V);
                // *** gl = dm * gl
                mul_v<ArchType::ASCEND_V220, float>(gl32_ubuf_tensor,
                    dm32_ubuf_tensor,
                    gl32_ubuf_tensor,
                    sub_m_d64,  // repeat
                    1,          // dstBlockStride
                    1,          // src0BlockStride
                    1,          // src1BlockStride
                    8,          // dstRepeatStride
                    8,          // src0RepeatStride
                    8           // src1RepeatStride
                );
                PIPE_BARRIER(V);
                // *** gl = ll + gl
                add_v<ArchType::ASCEND_V220, float>(gl32_ubuf_tensor,
                    gl32_ubuf_tensor,
                    ll_ubuf_tensor,
                    sub_m_d64,  // repeat
                    1,          // dstBlockStride
                    1,          // src0BlockStride
                    1,          // src1BlockStride
                    8,          // dstRepeatStride
                    8,          // src0RepeatStride
                    8           // src1RepeatStride
                );
                PIPE_BARRIER(V);
            }
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            brcb_v<ArchType::ASCEND_V220, uint32_t>(tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                dm32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                1,               // dstBlockStride
                8,               // dstRepeatStride
                round_sub_m / FLOAT_BLOCK_SIZE  // repeat
            );
            PIPE_BARRIER(V);
            if (head_loop > 1) {
                gm_to_ub<ArchType::ASCEND_V220, float>(
                    go32_ubuf_tensor,
                    go_gm_tensor,
                    0,
                    1,
                    sub_m * round_v / FLOAT_BLOCK_SIZE,
                    0,
                    0
                );
                SET_FLAG(MTE2, V, EVENT_ID0);
                WAIT_FLAG(MTE2, V, EVENT_ID0);
            }

            // *** go = go * dm_block
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < __v / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                mul_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,        // repeat
                    1,            // dstBlockStride
                    1,            // src0BlockStride
                    0,            // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1             // src1RepeatStride
                );
            }
            if (__v % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(__v % FLOAT_VECTOR_SIZE);
                mul_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,        // repeat
                    1,            // dstBlockStride
                    1,            // src0BlockStride
                    0,            // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1             // src1RepeatStride
                );
                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            PIPE_BARRIER(V);
            // *** go = lo + go
            add_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor,
                go32_ubuf_tensor,
                lo_ubuf_tensor,
                (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
                1,                            // dstBlockStride
                1,                            // src0BlockStride
                1,                            // src1BlockStride
                8,                            // dstRepeatStride
                8,                            // src0RepeatStride
                8                             // src1RepeatStride
            );
            PIPE_BARRIER(V);
        } else {
            // *** gl = ll
            if (head_loop_idx == 0) {
                ub_to_ub<ArchType::ASCEND_V220, float>(
                    gl32_ubuf_tensor,
                    ll_ubuf_tensor,
                    0,                // sid
                    1,                // nBurst
                    64 / FLOAT_BLOCK_SIZE,  // lenBurst
                    // round_sub_m / FLOAT_BLOCK_SIZE,  // lenBurst
                    0,                // srcGap
                    0                 // dstGap
                    );
                PIPE_BARRIER(V);
            }

            gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
                go32_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
                o_tmp_gm_tensor,
                0,                    // sid
                1,                    // nBurst
                sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                    // srcGap
                0                     // dstGap
            );
            if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
                DeQuantPerHeadImpl(
                    deq_scale_gm_tensor_k1[head_idx],
                    o_tmp_gm_tensor,
                    go32_ubuf_tensor, go32_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
                    descale_k1_ubuf_tensor, tv32_ubuf_tensor, pm32_ubuf_tensor, sub_m, round_v, round_v, 1, 0);
            } else {
                SET_FLAG(MTE2, V, EVENT_ID0);
                WAIT_FLAG(MTE2, V, EVENT_ID0);
            }
        }
        SET_FLAG(V, MTE2, EVENT_ID0);

        if (n_idx == n_loop - 1) {
            // *** gl_block = expand_to_block(gl)
            brcb_v<ArchType::ASCEND_V220, uint32_t>(tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                gl32_ubuf_tensor.ReinterpretCast<uint32_t>()[head_loop_idx * 16],
                1,               // dstBlockStride
                8,               // dstRepeatStride
                round_sub_m / FLOAT_BLOCK_SIZE  // repeat
            );
            PIPE_BARRIER(V);
            // *** go = go / gl_block
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vdiv_idx = 0; vdiv_idx < __v / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                div_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[vdiv_idx * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[vdiv_idx * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,                 // repeat
                    1,                     // dstBlockStride
                    1,                     // src0BlockStride
                    0,                     // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1                      // src1RepeatStride
                );
            }
            if (__v % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(__v % FLOAT_VECTOR_SIZE);
                div_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,                 // repeat
                    1,                     // dstBlockStride
                    1,                     // src0BlockStride
                    0,                     // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1                      // src1RepeatStride
                );
                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // fix hidden_size=96
            }
            PIPE_BARRIER(V);

            // *** go = castfp32to16(go)
            conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(go_ubuf_tensor,
                go32_ubuf_tensor,
                (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
                1,                            // dstBlockStride
                1,                            // srcBlockStride
                4,                            // dstRepeatStride
                8                             // srcRepeatStride
            );
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);

            uint32_t inner_o_gm_offset = 0;
            uint32_t inner_go_ubuf_offset = 0;

            if (head_res_row_num != 0) {
                AscendC::DataCopyPad(
                    o_gm_tensor[inner_o_gm_offset + q_heads * __v * head_start_sblock_idx],
                    go_ubuf_tensor[inner_go_ubuf_offset],
                    AscendC::DataCopyExtParams(
                        head_res_row_num,  // blockCount
                        __v * 2,    // blockLen
                        0,          // srcStride
                        __v * (q_heads - 1) * 2,  // dstStride
                        0           // rsv
                    )
                );
                inner_o_gm_offset += __v;
                inner_go_ubuf_offset += head_res_row_num * __v;
            }

            for (uint32_t i = 0; i < numhead_per_process; i++) {
                AscendC::DataCopyPad(
                    o_gm_tensor[inner_o_gm_offset],
                    go_ubuf_tensor[inner_go_ubuf_offset],
                    AscendC::DataCopyExtParams(
                        q_seq_len,  // blockCount
                        __v * 2,    // blockLen
                        0,          // srcStride
                        __v * (q_heads - 1) * 2,  // dstStride
                        0           // rsv
                    )
                );
                inner_o_gm_offset += __v;
                inner_go_ubuf_offset += q_seq_len * __v;
            }

            if (tail_res_row_num != 0) {
                AscendC::DataCopyPad(
                    o_gm_tensor[inner_o_gm_offset],
                    go_ubuf_tensor[inner_go_ubuf_offset],
                    AscendC::DataCopyExtParams(
                        tail_res_row_num,  // blockCount
                        __v * 2,    // blockLen
                        0,          // srcStride
                        __v * (q_heads - 1) * 2,  // dstStride
                        0           // rsv
                    )
                );
            }
            // ********************* move O to GM ************************
            if constexpr (IS_RING) {
                uint32_t lenBurst = sizeof(OUT_DTYPE);
                ln_v<ArchType::ASCEND_V220, float>(lse32_ubuf_tensor,
                    gl32_ubuf_tensor,
                    sub_m_d64,  // repeat
                    1,          // dstBlockStride
                    1,          // srcBlockStride
                    8,          // dstRepeatStride
                    8           // srcRepeatStride
                );
                PIPE_BARRIER(V);
                add_v<ArchType::ASCEND_V220, float>(lse32_ubuf_tensor,
                    lse32_ubuf_tensor,
                    gm32_ubuf_tensor,
                    sub_m_d64,  // repeat
                    1,          // dstBlockStride
                    1,          // src0BlockStride
                    1,          // src1BlockStride
                    8,          // dstRepeatStride
                    8,          // src0RepeatStride
                    8           // src1RepeatStride
                );
                PIPE_BARRIER(V);
                conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(lse_conv_ubuf_tensor,
                    lse32_ubuf_tensor,
                    sub_m_d64,                    // repeat
                    1,                            // dstBlockStride
                    1,                            // srcBlockStride
                    4,                            // dstRepeatStride
                    8                             // srcRepeatStride
                );
                SET_FLAG(V, MTE3, EVENT_ID1);
                WAIT_FLAG(V, MTE3, EVENT_ID1);
                // copyout lse
                ub_to_gm_align<ArchType::ASCEND_V220, OUT_DTYPE>(
                    lse_gm_tensor[(int64_t)(o_offset / __k)],
                    lse_conv_ubuf_tensor,
                    0,                 // sid
                    1,                 // nBurst
                    lenBurst * sub_m * head_loop,  // lenBurst
                    0,                 // leftPaddingNum
                    0,                 // rightPaddingNum
                    0,                 // srcGap
                    0                  // dstGap
                );
                SET_FLAG(MTE3, V, EVENT_ID1);
                WAIT_FLAG(MTE3, V, EVENT_ID1);
            }

        } else if (head_loop > 1) {
            SET_FLAG(V, MTE3, EVENT_ID5);
            WAIT_FLAG(V, MTE3, EVENT_ID5);
            ub_to_gm<ArchType::ASCEND_V220, float>(
                go_gm_tensor,
                go32_ubuf_tensor,
                0,
                1,
                sub_m * round_v / FLOAT_BLOCK_SIZE,
                0,
                0
            );
        }
        SET_FLAG(MTE3, MTE2, EVENT_ID4);
    }

    __aicore__ __attribute__((always_inline)) inline void SoftmaxStage2MLAHeadLoopTP1(
        AscendC::GlobalTensor<mm2CopyType> o_tmp_gm_tensor,
        AscendC::GlobalTensor<float> go_gm_tensor,
        AscendC::GlobalTensor<OUT_DTYPE> o_gm_tensor,
        AscendC::LocalTensor<float> dm32_ubuf_tensor,
        AscendC::LocalTensor<float> ll_ubuf_tensor,
        AscendC::LocalTensor<float> pm32_ubuf_tensor,
        uint32_t n_idx,
        uint32_t n_loop,
        uint32_t qk_n,
        uint32_t qk_round_n,
        uint32_t sub_m,
        uint64_t o_offset,
        uint32_t head_idx,
        uint32_t pm_flag_scalar,
        uint32_t head_loop,
        uint32_t head_loop_idx,
        uint32_t q_seq_len,
        uint32_t sub_head_num,
        uint32_t cur_head_num,
        uint32_t numhead_per_process,
        uint32_t head_res_row_num,
        uint32_t head_start_sblock_idx,
        uint32_t tail_res_row_num
        )
    {
        uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 64
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        if (n_idx != 4) {
            gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
                lo_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
                o_tmp_gm_tensor,
                0,                    // sid
                1,                    // nBurst
                sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                    // srcGap
                0                     // dstGap
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID4);
        if (n_idx != 4) {
            // expand_to_block
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            brcb_v<ArchType::ASCEND_V220, uint32_t>(tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                dm32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                1,               // dstBlockStride
                8,               // dstRepeatStride
                round_sub_m / FLOAT_BLOCK_SIZE  // repeat
            );
            PIPE_BARRIER(V);
            if (head_loop > 1) {
                gm_to_ub<ArchType::ASCEND_V220, float>(
                    go32_ubuf_tensor,
                    go_gm_tensor,
                    0,
                    1,
                    sub_m * round_v / FLOAT_BLOCK_SIZE,
                    0,
                    0
                );
                SET_FLAG(MTE2, V, EVENT_ID0);
                WAIT_FLAG(MTE2, V, EVENT_ID0);
            }

            // *** go = go * dm_block
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < __v / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                mul_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,        // repeat
                    1,            // dstBlockStride
                    1,            // src0BlockStride
                    0,            // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1             // src1RepeatStride
                );
            }
            if (__v % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(__v % FLOAT_VECTOR_SIZE);
                mul_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,        // repeat
                    1,            // dstBlockStride
                    1,            // src0BlockStride
                    0,            // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1             // src1RepeatStride
                );
                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            PIPE_BARRIER(V);
            // *** go = lo + go
            add_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor,
                go32_ubuf_tensor,
                lo_ubuf_tensor,
                (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
                1,                            // dstBlockStride
                1,                            // src0BlockStride
                1,                            // src1BlockStride
                8,                            // dstRepeatStride
                8,                            // src0RepeatStride
                8                             // src1RepeatStride
            );
            PIPE_BARRIER(V);
        } else {
            // *** go = lo

            gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
                go32_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
                o_tmp_gm_tensor,
                0,                    // sid
                1,                    // nBurst
                sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                    // srcGap
                0                     // dstGap
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }
        SET_FLAG(V, MTE2, EVENT_ID0);

        if (n_idx + 4 > n_loop + 4 - 1) {
            // *** gl_block = expand_to_block(gl), 存放于 tv
            brcb_v<ArchType::ASCEND_V220, uint32_t>(tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                gl32_ubuf_tensor.ReinterpretCast<uint32_t>()[head_loop_idx * 16],
                1,               // dstBlockStride
                8,               // dstRepeatStride
                round_sub_m / FLOAT_BLOCK_SIZE  // repeat
            );
            PIPE_BARRIER(V);
            // *** go = go / gl_block
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vdiv_idx = 0; vdiv_idx < __v / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                div_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[vdiv_idx * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[vdiv_idx * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,                 // repeat
                    1,                     // dstBlockStride
                    1,                     // src0BlockStride
                    0,                     // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1                      // src1RepeatStride
                );
            }
            if (__v % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(__v % FLOAT_VECTOR_SIZE);
                div_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,                 // repeat
                    1,                     // dstBlockStride
                    1,                     // src0BlockStride
                    0,                     // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1                      // src1RepeatStride
                );
                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // fix hidden_size=96
            }
            PIPE_BARRIER(V);

            // *** go = castfp32to16(go)
            conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(go_ubuf_tensor,
                go32_ubuf_tensor,
                (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
                1,                            // dstBlockStride
                1,                            // srcBlockStride
                4,                            // dstRepeatStride
                8                             // srcRepeatStride
            );
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);

            uint32_t inner_o_gm_offset = 0;
            uint32_t inner_go_ubuf_offset = 0;

            if (head_res_row_num != 0) {
                AscendC::DataCopyPad(
                    o_gm_tensor[inner_o_gm_offset + q_heads * __v * head_start_sblock_idx],
                    go_ubuf_tensor[inner_go_ubuf_offset],
                    AscendC::DataCopyExtParams(
                        head_res_row_num,  // blockCount
                        __v * 2,    // blockLen
                        0,          // srcStride
                        __v * (q_heads - 1) * 2,  // dstStride
                        0           // rsv
                    )
                );
                inner_o_gm_offset += __v;
                inner_go_ubuf_offset += head_res_row_num * __v;
            }

            for (uint32_t i = 0; i < numhead_per_process; i++) {
                AscendC::DataCopyPad(
                    o_gm_tensor[inner_o_gm_offset],
                    go_ubuf_tensor[inner_go_ubuf_offset],
                    AscendC::DataCopyExtParams(
                        q_seq_len,  // blockCount
                        __v * 2,    // blockLen
                        0,          // srcStride
                        __v * (q_heads - 1) * 2,  // dstStride
                        0           // rsv
                    )
                );
                inner_o_gm_offset += __v;
                inner_go_ubuf_offset += q_seq_len * __v;
            }

            if (tail_res_row_num != 0) {
                AscendC::DataCopyPad(
                    o_gm_tensor[inner_o_gm_offset],
                    go_ubuf_tensor[inner_go_ubuf_offset],
                    AscendC::DataCopyExtParams(
                        tail_res_row_num,  // blockCount
                        __v * 2,    // blockLen
                        0,          // srcStride
                        __v * (q_heads - 1) * 2,  // dstStride
                        0           // rsv
                    )
                );
            }
            // ********************* move O to GM ************************
            if constexpr (IS_RING) {
                uint32_t lenBurst = sizeof(OUT_DTYPE);
                ln_v<ArchType::ASCEND_V220, float>(lse32_ubuf_tensor,
                    gl32_ubuf_tensor,
                    sub_m_d64,  // repeat
                    1,          // dstBlockStride
                    1,          // srcBlockStride
                    8,          // dstRepeatStride
                    8           // srcRepeatStride
                );
                PIPE_BARRIER(V);
                add_v<ArchType::ASCEND_V220, float>(lse32_ubuf_tensor,
                    lse32_ubuf_tensor,
                    gm32_ubuf_tensor,
                    sub_m_d64,  // repeat
                    1,          // dstBlockStride
                    1,          // src0BlockStride
                    1,          // src1BlockStride
                    8,          // dstRepeatStride
                    8,          // src0RepeatStride
                    8           // src1RepeatStride
                );
                PIPE_BARRIER(V);
                conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(lse_conv_ubuf_tensor,
                    lse32_ubuf_tensor,
                    sub_m_d64,                    // repeat
                    1,                            // dstBlockStride
                    1,                            // srcBlockStride
                    4,                            // dstRepeatStride
                    8                             // srcRepeatStride
                );
                SET_FLAG(V, MTE3, EVENT_ID1);
                WAIT_FLAG(V, MTE3, EVENT_ID1);
                // copyout lse
                ub_to_gm_align<ArchType::ASCEND_V220, OUT_DTYPE>(
                    lse_gm_tensor[(int64_t)(o_offset / __k)],
                    lse_conv_ubuf_tensor,
                    0,                 // sid
                    1,                 // nBurst
                    lenBurst * sub_m * head_loop,  // lenBurst
                    0,                 // leftPaddingNum
                    0,                 // rightPaddingNum
                    0,                 // srcGap
                    0                  // dstGap
                );
                SET_FLAG(MTE3, V, EVENT_ID1);
                WAIT_FLAG(MTE3, V, EVENT_ID1);
            }
        }
        else if (head_loop > 1) {
            SET_FLAG(V, MTE3, EVENT_ID5);
            WAIT_FLAG(V, MTE3, EVENT_ID5);
            ub_to_gm<ArchType::ASCEND_V220, float>(
                go_gm_tensor,
                go32_ubuf_tensor,
                0,
                1,
                sub_m * round_v / FLOAT_BLOCK_SIZE,
                0,
                0
            );
        }
        SET_FLAG(MTE3, MTE2, EVENT_ID4);
    }

    __aicore__ __attribute__((always_inline)) inline void TailSoftmaxStage2MLAHeadLoopTP1(
        AscendC::GlobalTensor<mm2CopyType> o_tmp_gm_tensor,
        AscendC::GlobalTensor<float> go_gm_tensor,
        AscendC::GlobalTensor<float> gl_gm_tensor,
        AscendC::GlobalTensor<float> gm_gm_tensor,
        AscendC::LocalTensor<float> dm32_ubuf_tensor,
        AscendC::LocalTensor<float> go32_ubuf_tensor,
        AscendC::LocalTensor<float> gl32_ubuf_tensor,
        AscendC::LocalTensor<float> gm32_ubuf_tensor,
        uint32_t n_idx,
        uint32_t n_loop,
        uint32_t qk_n,
        uint32_t qk_round_n,
        uint32_t sub_m,
        uint64_t o_offset,
        uint32_t head_idx,
        uint32_t head_loop,
        uint32_t head_loop_idx,
        uint32_t q_seq_len,
        uint32_t sub_head_num,
        uint32_t cur_head_num,
        uint32_t numhead_per_process
        )
    {
        uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 64
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        if (n_idx != 4) {
            gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
                lo_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
                o_tmp_gm_tensor,
                0,                    // sid
                1,                    // nBurst
                sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                    // srcGap
                0                     // dstGap
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID4);
        if (n_idx != 4) {
            // expand_to_block
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            brcb_v<ArchType::ASCEND_V220, uint32_t>(tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                dm32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                1,               // dstBlockStride
                8,               // dstRepeatStride
                round_sub_m / FLOAT_BLOCK_SIZE  // repeat
            );
            PIPE_BARRIER(V);
            if (head_loop > 1) {
                gm_to_ub<ArchType::ASCEND_V220, float>(
                    go32_ubuf_tensor,
                    go_gm_tensor,
                    0,
                    1,
                    sub_m * round_v / FLOAT_BLOCK_SIZE,
                    0,
                    0
                );
                SET_FLAG(MTE2, V, EVENT_ID0);
                WAIT_FLAG(MTE2, V, EVENT_ID0);
            }

            // *** go = go * dm_block
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < __v / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                mul_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,        // repeat
                    1,            // dstBlockStride
                    1,            // src0BlockStride
                    0,            // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1             // src1RepeatStride
                );
            }
            if (__v % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(__v % FLOAT_VECTOR_SIZE);
                mul_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,        // repeat
                    1,            // dstBlockStride
                    1,            // src0BlockStride
                    0,            // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1             // src1RepeatStride
                );
                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            PIPE_BARRIER(V);
            // *** go = lo + go
            add_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor,
                go32_ubuf_tensor,
                lo_ubuf_tensor,
                (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
                1,                            // dstBlockStride
                1,                            // src0BlockStride
                1,                            // src1BlockStride
                8,                            // dstRepeatStride
                8,                            // src0RepeatStride
                8                             // src1RepeatStride
            );
            PIPE_BARRIER(V);
        } else {
            // *** go = lo

            gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
                go32_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
                o_tmp_gm_tensor,
                0,                    // sid
                1,                    // nBurst
                sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                    // srcGap
                0                     // dstGap
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }
        SET_FLAG(V, MTE2, EVENT_ID0);

        if (n_idx + 4 > n_loop + 4 - 1) {
            // The last step to process the o with dividing and copyout
            // TODO: Maybe the following two don't need waiting
            // Copyout gl32_ubuf_tensor to gl_gm_tensor
            ub_to_gm<ArchType::ASCEND_V220, float>(
                gl_gm_tensor,
                gl32_ubuf_tensor,
                0,
                1,
                sub_m / FLOAT_BLOCK_SIZE,
                0,
                0
            );
            
            // Copyout rowmax to global gm32_ubuf_tensor
            ub_to_gm<ArchType::ASCEND_V220, float>(
                gm_gm_tensor,
                gm32_ubuf_tensor,
                0,
                1,
                sub_m / FLOAT_BLOCK_SIZE,
                0,
                0
            );
            // Copyout go32_ubuf_tensor to go_gm_tensor
            // This is needed to wait for former calculation
            SET_FLAG(V, MTE3, EVENT_ID5);
            WAIT_FLAG(V, MTE3, EVENT_ID5);
            ub_to_gm<ArchType::ASCEND_V220, float>(
                go_gm_tensor,
                go32_ubuf_tensor,
                0,
                1,
                sub_m * round_v / FLOAT_BLOCK_SIZE,
                0,
                0
            );
            
        }
        else if (head_loop > 1) {
            SET_FLAG(V, MTE3, EVENT_ID5);
            WAIT_FLAG(V, MTE3, EVENT_ID5);
            ub_to_gm<ArchType::ASCEND_V220, float>(
                go_gm_tensor,
                go32_ubuf_tensor,
                0,
                1,
                sub_m * round_v / FLOAT_BLOCK_SIZE,
                0,
                0
            );
        }
        SET_FLAG(MTE3, MTE2, EVENT_ID4);
        PIPE_BARRIER(ALL);
    }

    __aicore__ __attribute__((always_inline)) inline void InnerRunVectorChange(
        uint32_t cur_batch, uint32_t start_head, uint32_t cur_nIndx,
        uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t cur_head_num,
        uint32_t offset_tiling, uint32_t embed_split_size_v, uint32_t embed_split_loop_v)
    {
        uint32_t addr_o_high32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 4 + offset_tiling));
        uint32_t addr_o_loww32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 5 + offset_tiling));
        uint64_t addr_o_scalar = (uint64_t)(((uint64_t)addr_o_high32) << 32 | addr_o_loww32);

        uint32_t addr_mask_high32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 6 + offset_tiling));
        uint32_t addr_mask_loww32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 7 + offset_tiling));
        uint64_t addr_mask_scalar = (uint64_t)(((uint64_t)addr_mask_high32) << 32 | addr_mask_loww32);

        uint32_t mask_offset = addr_mask_scalar;

        uint32_t pp_n_scalar = block_size; // 64
        uint32_t sub_n_loop = pp_n_scalar / block_size;
        uint32_t real_n_loop = (cur_kv_seqlen + block_size - 1) / block_size;

        uint32_t n_loop = (cur_kv_seqlen + pp_n_scalar - 1) / pp_n_scalar;

        uint32_t qk_n = pp_n_scalar;
        uint32_t qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);

        uint32_t qk_n_2 = pp_n_scalar;
        uint32_t qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);

        // split head num to two vectors
        uint32_t sub_head_num = (sub_block_idx == 1) ? (cur_head_num - cur_head_num / 2) : cur_head_num / 2; // 16
        uint32_t sub_m = sub_head_num * cur_q_seqlen; // 16 * 3 = 48

        uint32_t head_idx = (sub_block_idx == 0) ? start_head : start_head + cur_head_num / 2 * cur_q_seqlen; // not used

        o_offset = addr_o_scalar + start_head * embedding_size + sub_block_idx * cur_head_num / 2 * embedding_size; // for NSD -> SND

        uint32_t sub_m_d128 = (sub_m + 127) / 128;  // up aligned to 128
        uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 128
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;

        uint32_t start_kv = 0;
        /* if tail length smalller than q_len - 1, then need to mask the last two tile*/
        uint32_t tail_len = cur_kv_seqlen - (n_loop - 1) * pp_n_scalar;
        bool prev_tail_mask = (n_loop > 1 && tail_len < cur_q_seqlen - 1);
        for (uint32_t n_idx = 0; n_idx < n_loop + 1; n_idx++) {
            if (n_idx != n_loop) {
                bool need_mask = false;
                uint32_t mask_start_offset = 0;
                if (n_idx == (n_loop - 2)) {
                    need_mask = prev_tail_mask;
                    mask_start_offset = need_mask ? (tail_len + MASK_COLUMNS - 1) * MASK_COLUMNS : 0;
                }
                if (n_idx == (n_loop - 1)) {
                    qk_n = (cur_kv_seqlen - n_idx * pp_n_scalar);
                    qk_round_n = RoundUp<16>(qk_n);
                    need_mask = true;
                    mask_start_offset = (qk_n - 1) * MASK_COLUMNS;
                }
                WaitFlagDev(QK_READY_DECODER);
                /* ************ softmax1 stage1  ************* */
                WAIT_FLAG(MTE3, MTE2, EVENT_ID3);
                if (sub_m > 0) {
                    if (mask_type == 3) {
                        mask_start_offset = mask_offset + n_idx * pp_n_scalar;
                    }
                    // input QK shape (sub_m, qk_round_n)
                    if (n_idx % 2 == 0){
                        SoftmaxStage1(
                            p_gm_tensor[(uint64_t)block_idx * TMP_SIZE * T_BLOCK_OFFSET +
                                (uint64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n * T_BLOCK_OFFSET + (uint64_t)(n_idx % 2) * TMP_SIZE * T_BLOCK_OFFSET / 2],
                            s_gm_tensor[(int64_t)block_idx * TMP_SIZE_DECODER +
                                (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n + (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
                            s_rope_gm_tensor[(int64_t)block_idx * TMP_SIZE_DECODER +
                                (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n + (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
                            mask_gm_tensor[mask_start_offset],
                            dm32_ubuf_tensor, ll_ubuf_tensor, pm32_ubuf_tensor,
                            n_idx, qk_n, qk_round_n, sub_m, 0, sub_n_loop, cur_batch, start_kv, real_n_loop, head_idx, pm_flag_scalar1, cur_q_seqlen, cur_kv_seqlen, need_mask
                        );
                    } else {
                        SoftmaxStage1(
                            p_gm_tensor[(uint64_t)block_idx * TMP_SIZE * T_BLOCK_OFFSET  +
                                (uint64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n * T_BLOCK_OFFSET +
                                TMP_SIZE * T_BLOCK_OFFSET / 2],
                            s_gm_tensor[(int64_t)block_idx * TMP_SIZE_DECODER +
                                (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n +
                                TMP_SIZE_DECODER / 2],
                            s_rope_gm_tensor[(int64_t)block_idx * TMP_SIZE_DECODER +
                                (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n +
                                TMP_SIZE_DECODER / 2],
                            mask_gm_tensor[mask_start_offset],
                            dm32_stage2_ubuf_tensor, ll_stage2_ubuf_tensor, pm32_ubuf_stage2_tensor,
                            n_idx, qk_n, qk_round_n, sub_m, 0, sub_n_loop, cur_batch, start_kv, real_n_loop, head_idx, pm_flag_scalar2, cur_q_seqlen, cur_kv_seqlen, need_mask
                        );
                    }
                }
                FftsCrossCoreSync<PIPE_MTE3, 2>(SOFTMAX_READY_DECODER);

                SET_FLAG(MTE3, MTE2, EVENT_ID3);
            }
            /* ************ softmax2 stage1  ************* */

            uint32_t process_row_num = 16;
            uint32_t numhead_per_process = process_row_num / cur_q_seqlen;
            if (n_idx != 0) {
                if (n_idx == n_loop) {
                    qk_n_2 = (cur_kv_seqlen - (n_idx - 1) * pp_n_scalar);
                    qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);
                }
                WaitFlagDev(UPDATE_READY_DECODER);
                if (sub_m > 0) {
                    uint32_t head_loop = (sub_m + process_row_num - 1) / process_row_num;

                    uint32_t head_res_row_num = 0;
                    uint32_t head_start_sblock_idx = 0;
                    uint32_t tail_res_row_num = 0;

                    for (uint32_t head_loop_idx = 0; head_loop_idx < head_loop; ++head_loop_idx) {
                        uint32_t head_offset = head_loop_idx * process_row_num * round_v;
                        uint32_t cur_sub_m = head_loop_idx == (head_loop - 1) ? sub_m - head_loop_idx * process_row_num : process_row_num; // 15 or 3

                        // complete head num
                        head_start_sblock_idx = tail_res_row_num;
                        head_res_row_num = (cur_q_seqlen - tail_res_row_num) % cur_q_seqlen;
                        uint32_t cur_numhead_per_process = (cur_sub_m - head_res_row_num) / cur_q_seqlen;
                        tail_res_row_num = cur_sub_m - cur_numhead_per_process * cur_q_seqlen - head_res_row_num;

                        uint32_t out_o_offset = head_loop_idx * numhead_per_process * round_v; // modified, round_v = 512

                        SoftmaxStage2MLAHeadLoop(
                            o_tmp_gm_tensor[(uint64_t)(block_idx * TMP_SIZE * 2 + sub_block_idx * cur_head_num * cur_q_seqlen / 2 * round_v + head_offset + ((n_idx - 1) % 2) * TMP_SIZE)],
                            go_gm_tensor[(uint64_t)(block_idx * TMP_SIZE + sub_block_idx * cur_head_num * cur_q_seqlen / 2 * round_v + head_offset)],
                            o_gm_tensor[(uint64_t)(o_offset + out_o_offset)],
                            dm32_ubuf_tensor[(uint64_t)((n_idx - 1) % 2 * 128 + head_loop_idx * process_row_num)],
                            ll_ubuf_tensor[(uint64_t)((n_idx - 1) % 2 * 256 + head_loop_idx * process_row_num)],
                            pm32_ubuf_tensor[(uint64_t)((n_idx - 1) % 2 * 128 + head_loop_idx * process_row_num)],
                            n_idx - 1, n_loop, qk_n_2, RoundUp<T_BLOCK_SIZE>(qk_round_n_2), cur_sub_m, o_offset,
                            head_idx + head_loop_idx * process_row_num,
                            pm_flag_scalar1, head_loop, head_loop_idx, cur_q_seqlen, sub_head_num, cur_head_num,
                            cur_numhead_per_process,
                            head_res_row_num, head_start_sblock_idx, tail_res_row_num);
                    }
                }
            }
        }
    }

    __aicore__ __attribute((always_inline)) inline void SoftmaxGatherTP1(
        AscendC::GlobalTensor<OUT_DTYPE> o_gm_tensor,
        AscendC::GlobalTensor<float> go_gm_tensor,
        AscendC::GlobalTensor<float> gl_gm_tensor,
        AscendC::GlobalTensor<float> gm_gm_tensor,
        AscendC::LocalTensor<OUT_DTYPE> go_ubuf_tensor,
        AscendC::LocalTensor<float> go32_ubuf_tensor,
        AscendC::LocalTensor<float> gl32_ubuf_tensor,
        AscendC::LocalTensor<float> gm32_ubuf_tensor,
        AscendC::LocalTensor<float> lo_ubuf_tensor,
        AscendC::LocalTensor<float> ll_ubuf_tensor,
        AscendC::LocalTensor<float> lm32_ubuf_tensor,
        AscendC::LocalTensor<float> hm32_ubuf_tensor,
        AscendC::LocalTensor<float> dm32_ubuf_tensor,
        AscendC::LocalTensor<float> tv32_ubuf_tensor,
        uint32_t start_core_idx,
        uint32_t cores_process,
        uint32_t cur_core_idx,
        uint32_t cur_process_row
        )
    {
        if (cur_core_idx == start_core_idx) {
            // just copyin
            gm_to_ub<ArchType::ASCEND_V220, float>(
                go32_ubuf_tensor,
                go_gm_tensor,
                0,
                1,
                cur_process_row * round_v / FLOAT_BLOCK_SIZE,
                0,
                0
            );
            
            gm_to_ub<ArchType::ASCEND_V220, float>(
                gl32_ubuf_tensor,
                gl_gm_tensor,
                0,
                1,
                cur_process_row / FLOAT_BLOCK_SIZE,
                0,
                0
            );
            
            gm_to_ub<ArchType::ASCEND_V220, float>(
                gm32_ubuf_tensor,
                gm_gm_tensor,
                0,
                1,
                cur_process_row / FLOAT_BLOCK_SIZE,
                0,
                0
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }
        else{
            gm_to_ub<ArchType::ASCEND_V220, float>(
                lo_ubuf_tensor,
                go_gm_tensor,
                0,
                1,
                cur_process_row * round_v / FLOAT_BLOCK_SIZE,
                0,
                0
            );
            
            gm_to_ub<ArchType::ASCEND_V220, float>(
                ll_ubuf_tensor,
                gl_gm_tensor,
                0,
                1,
                cur_process_row / FLOAT_BLOCK_SIZE,
                0,
                0
            );
            
            gm_to_ub<ArchType::ASCEND_V220, float>(
                lm32_ubuf_tensor,
                gm_gm_tensor,
                0,
                1,
                cur_process_row / FLOAT_BLOCK_SIZE,
                0,
                0
            );
            
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }
        if (cur_core_idx != start_core_idx) {
            // * update the o
            // * hm = max(gm, lm)
            AscendC::Max<float>(
                hm32_ubuf_tensor,
                lm32_ubuf_tensor,
                gm32_ubuf_tensor,
                cur_process_row
            );
            AscendC::PipeBarrier<PIPE_V>();

            // ** update go with dm
            // ** dm = gm - hm
            AscendC::Sub<float>(
                dm32_ubuf_tensor,
                gm32_ubuf_tensor,
                hm32_ubuf_tensor,
                cur_process_row
            );
            // ** update lo with dm
            // ** dm = lm - hm
            AscendC::Sub<float>(
                dm32_ubuf_tensor[cur_process_row],
                lm32_ubuf_tensor,
                hm32_ubuf_tensor,
                cur_process_row
            );
            AscendC::PipeBarrier<PIPE_V>();

            // ** dm = exp(dm)
            AscendC::Exp<float>(
                dm32_ubuf_tensor,
                dm32_ubuf_tensor,
                cur_process_row * 2
            );
            AscendC::PipeBarrier<PIPE_V>();

            // ** gl = gl * dm
            AscendC::Mul<float, true>(
                gl32_ubuf_tensor,
                dm32_ubuf_tensor,
                gl32_ubuf_tensor,
                (uint64_t)cur_process_row,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8)
            );
            // ** ll = ll * dm
            AscendC::Mul<float, true>(
                ll_ubuf_tensor,
                dm32_ubuf_tensor[cur_process_row],
                ll_ubuf_tensor,
                (uint64_t)cur_process_row, //mask
                1, // repeat 
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8)
            );
            
            // ** dm_block = expand_to_block(dm)
            AscendC::Brcb<float>(
                tv32_ubuf_tensor,
                dm32_ubuf_tensor,
                2 * cur_process_row / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8)
            );
            AscendC::PipeBarrier<PIPE_V>();
            
            // * gl = gl + ll
            AscendC::Add<float, true>(
                gl32_ubuf_tensor,
                ll_ubuf_tensor,
                gl32_ubuf_tensor,
                (uint64_t)cur_process_row,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8)
            );

            // ** go = go * dm_block
            for (uint32_t vmul_idx = 0; vmul_idx < __v / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                AscendC::Mul<float, true>(
                    go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    (uint64_t)FLOAT_VECTOR_SIZE,
                    cur_process_row,
                    AscendC::BinaryRepeatParams(1, 1, 0, round_v / FLOAT_BLOCK_SIZE, round_v / FLOAT_BLOCK_SIZE, 1)
                );
            }
            // ** lo = lo * dm_block
            for (uint32_t vmul_idx = 0; vmul_idx < __v / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                AscendC::Mul<float, true>(
                    lo_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                    lo_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor[cur_process_row * FLOAT_BLOCK_SIZE],
                    (uint64_t)FLOAT_VECTOR_SIZE,
                    cur_process_row,
                    AscendC::BinaryRepeatParams(1, 1, 0, round_v / FLOAT_BLOCK_SIZE, round_v / FLOAT_BLOCK_SIZE, 1)
                );
            }
            AscendC::PipeBarrier<PIPE_V>();
            
            
            // * go = go + lo
            AscendC::Add<float, true>(
                go32_ubuf_tensor,
                lo_ubuf_tensor,
                go32_ubuf_tensor,
                (uint64_t)FLOAT_VECTOR_SIZE,
                cur_process_row * round_v / FLOAT_VECTOR_SIZE,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8)
            );
            
            // gm = hm
            ub_to_ub<ArchType::ASCEND_V220, float>(
                gm32_ubuf_tensor,
                hm32_ubuf_tensor,
                0,
                1,
                cur_process_row / FLOAT_BLOCK_SIZE,
                0,
                0
            );
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (cur_core_idx == start_core_idx + cores_process - 1) {
            // gl_block = expand_to_block(gl)
            // AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb<float>(
                tv32_ubuf_tensor,
                gl32_ubuf_tensor,
                cur_process_row / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8)
            );
            AscendC::PipeBarrier<PIPE_V>();
            
            // go = go / gl_block
            for (uint32_t vmul_idx = 0; vmul_idx < __v / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                AscendC::Div<float, true>(
                    go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    (uint64_t)FLOAT_VECTOR_SIZE,
                    cur_process_row,
                    AscendC::BinaryRepeatParams(1, 1, 0, round_v / FLOAT_BLOCK_SIZE, round_v / FLOAT_BLOCK_SIZE, 1)
                );
            }
            AscendC::PipeBarrier<PIPE_V>();
            
            if constexpr (std::is_same<OUT_DTYPE, __bf16>::value) {
                AscendC::Cast<OUT_DTYPE, float, true>(
                    go_ubuf_tensor,
                    go32_ubuf_tensor,
                    AscendC::RoundMode::CAST_RINT, // not consistent with the document!
                    (uint64_t)FLOAT_VECTOR_SIZE,
                    cur_process_row * round_v / FLOAT_VECTOR_SIZE,
                    AscendC::UnaryRepeatParams(1, 1, 4, 8)
                );
            } else {
                AscendC::Cast<OUT_DTYPE, float, true>(
                    go_ubuf_tensor,
                    go32_ubuf_tensor,
                    AscendC::RoundMode::CAST_NONE,
                    (uint64_t)FLOAT_VECTOR_SIZE,
                    cur_process_row * round_v / FLOAT_VECTOR_SIZE,
                    AscendC::UnaryRepeatParams(1, 1, 4, 8)
                );
            }

            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            
            // copyout
            ub_to_gm<ArchType::ASCEND_V220, OUT_DTYPE>(
                o_gm_tensor,
                go_ubuf_tensor,
                0,
                cur_process_row,
                __v * sizeof(OUT_DTYPE) / BLOCK_SIZE_32,
                0,
                0
            );
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ __attribute__((always_inline)) inline void InnerRunVectorChangeTP1(
        uint32_t cur_batch, uint32_t start_head, uint32_t cur_nIndx,
        uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t cur_head_num,
        uint32_t offset_tiling, uint32_t embed_split_size_v, uint32_t embed_split_loop_v)
    {
        uint32_t prev_task = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 1 + offset_tiling));
        uint64_t addr_o_scalar = prev_task * q_heads * embedding_size;
        uint64_t addr_mask_scalar = 0;
        uint32_t mask_offset = addr_mask_scalar;

        uint32_t pp_n_scalar = block_size; // 64
        uint32_t sub_n_loop = pp_n_scalar / block_size;
        uint32_t real_n_loop = (cur_kv_seqlen + block_size - 1) / block_size;

        uint32_t n_loop = (cur_kv_seqlen + pp_n_scalar - 1) / pp_n_scalar;

        uint32_t qk_n = pp_n_scalar;
        uint32_t qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);

        uint32_t qk_n_2 = pp_n_scalar;
        uint32_t qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);

        // split head num to two vectors
        uint32_t sub_head_num = (sub_block_idx == 1) ? (cur_head_num - cur_head_num / 2) : cur_head_num / 2; // 16
        uint32_t sub_m = sub_head_num * cur_q_seqlen; // 16 * 3 = 48

        uint32_t head_idx = (sub_block_idx == 0) ? start_head : start_head + cur_head_num / 2 * cur_q_seqlen; // not used

        o_offset = addr_o_scalar + start_head * embedding_size + sub_block_idx * cur_head_num / 2 * embedding_size; // for NSD -> SND

        uint32_t sub_m_d128 = (sub_m + 127) / 128;  // up aligned to 128
        uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 128
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;

        uint32_t start_kv = 0;
        uint32_t s_block_stack = 4;
        uint32_t m_slice = FLOAT_VECTOR_SIZE / s_block_stack;
        uint32_t m_end = (sub_m + m_slice - 1) / m_slice;
        for (uint32_t n_idx = 0; n_idx < n_loop + s_block_stack; n_idx += s_block_stack) {
            if (n_idx < n_loop) {
                if (n_idx + s_block_stack > n_loop - 1) {
                    qk_n = (cur_kv_seqlen - n_idx * pp_n_scalar);
                } else {
                    qk_n =  pp_n_scalar * s_block_stack;
                }
                qk_round_n = RoundUp<16>(qk_n);
                if (sub_m == 0) {
                    WaitFlagDev(QK_READY_DECODER);
                }
                uint32_t pingpong_flag = 0;
                for (uint32_t m_ind = 0; m_ind < m_end; m_ind++) {
                    uint32_t row_offset = m_ind * m_slice;
                    uint32_t curr_m = m_ind == m_end - 1 ? sub_m - row_offset : m_slice;
                    uint32_t s_ub_offset = pingpong_flag * 8192;
                    uint32_t p_gm_offset = (uint64_t)block_idx * TMP_SIZE * 2 +
                                            (uint64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n + row_offset * qk_round_n + (uint64_t)((n_idx / s_block_stack) % 2) * TMP_SIZE;
                    uint32_t s_gm_offset = (int64_t)block_idx * TMP_SIZE_DECODER * 4 +
                                            (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n + row_offset * qk_round_n + (uint64_t)((n_idx / s_block_stack) % 2) * TMP_SIZE_DECODER * 2;
                    if (m_ind == 0) {
                        WaitFlagDev(QK_READY_DECODER);
                    }
                    if (curr_m == 0) {
                        continue;
                    }
                    OnlineSoftmaxStage1<float, float, IN_DTYPE, IN_DTYPE, MaskType::MASK_TYPE_NONE> (
                        ls32_ubuf_tensor[s_ub_offset],
                        mask_ubuf_tensor,
                        mask_ubuf_tensor.template ReinterpretCast<float>(),
                        lm32_ubuf_tensor[row_offset],
                        hm32_ubuf_tensor[row_offset],
                        gm32_ubuf_tensor[row_offset],
                        dm32_ubuf_tensor[((n_idx / s_block_stack) % 2) * UB_FLOAT_LINE_SIZE + row_offset],
                        ls32_ubuf_tensor[s_ub_offset],
                        ll_ubuf_tensor[row_offset],
                        gl32_ubuf_tensor[row_offset],
                        lp_ubuf_tensor[s_ub_offset * 2],
                        tv32_ubuf_tensor,
                        s_gm_tensor[s_gm_offset],
                        p_gm_tensor[p_gm_offset],
                        n_idx == 0, this->tor,
                        curr_m, qk_n, qk_round_n, pingpong_flag
                    );
                    pingpong_flag = 1 - pingpong_flag;
                }
                FftsCrossCoreSync<PIPE_MTE3, 2>(SOFTMAX_READY_DECODER);
            }
            /* ************ softmax2 stage1  ************* */
            // PIPE_BARRIER(ALL);
            uint32_t process_row_num = 16;
            uint32_t numhead_per_process = process_row_num / cur_q_seqlen;

            if (n_idx >= s_block_stack) {
                if (n_idx == n_loop) {
                    qk_n_2 = (cur_kv_seqlen - (n_idx - 1) * pp_n_scalar);
                    qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);
                }
                WaitFlagDev(UPDATE_READY_DECODER);
                if (sub_m > 0) {
                    uint32_t head_loop = (sub_m + process_row_num - 1) / process_row_num;

                    uint32_t head_res_row_num = 0;
                    uint32_t head_start_sblock_idx = 0;
                    uint32_t tail_res_row_num = 0;

                    for (uint32_t head_loop_idx = 0; head_loop_idx < head_loop; ++head_loop_idx) {
                        uint32_t head_offset = head_loop_idx * process_row_num * round_v;
                        uint32_t cur_sub_m = head_loop_idx == (head_loop - 1) ? sub_m - head_loop_idx * process_row_num : process_row_num; // 15 or 3

                        // complete head num
                        head_start_sblock_idx = tail_res_row_num;
                        head_res_row_num = (cur_q_seqlen - tail_res_row_num) % cur_q_seqlen;
                        uint32_t cur_numhead_per_process = (cur_sub_m - head_res_row_num) / cur_q_seqlen;
                        tail_res_row_num = cur_sub_m - cur_numhead_per_process * cur_q_seqlen - head_res_row_num;

                        uint32_t out_o_offset = head_loop_idx * numhead_per_process * round_v; // modified, round_v = 512

                        SoftmaxStage2MLAHeadLoopTP1(
                            o_tmp_gm_tensor[(uint64_t)(block_idx * TMP_SIZE * 2 + sub_block_idx * cur_head_num * cur_q_seqlen / 2 * round_v + head_offset + ((n_idx / s_block_stack - 1) % 2) * TMP_SIZE)],
                            go_gm_tensor[(uint64_t)(block_idx * TMP_SIZE + sub_block_idx * cur_head_num * cur_q_seqlen / 2 * round_v + head_offset)],
                            o_gm_tensor[(uint64_t)(o_offset + out_o_offset)],
                            dm32_ubuf_tensor[(uint64_t)((n_idx / s_block_stack - 1) % 2 * UB_FLOAT_LINE_SIZE + head_loop_idx * process_row_num)],
                            ll_ubuf_tensor[(uint64_t)((n_idx / s_block_stack - 1) % 2 * 256 + head_loop_idx * process_row_num)],
                            pm32_ubuf_tensor,
                            n_idx, n_loop, qk_n_2, RoundUp<T_BLOCK_SIZE>(qk_round_n_2), cur_sub_m, o_offset, head_idx,
                            pm_flag_scalar1, head_loop, head_loop_idx, cur_q_seqlen, sub_head_num, cur_head_num, cur_numhead_per_process,
                            head_res_row_num, head_start_sblock_idx, tail_res_row_num);
                    }
                }
            }
        }
    }

    __aicore__ __attribute__((always_inline)) inline void TailInnerRunVectorChangeTP1(
        uint32_t start_head,
        uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t cur_head_num,
        uint32_t offset_tiling, uint32_t embed_split_size_v, uint32_t embed_split_loop_v)
    {
        uint32_t prev_task = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 1 + offset_tiling));
        uint64_t addr_o_scalar = prev_task * q_heads * embedding_size;
        uint64_t addr_mask_scalar = 0;
        uint32_t mask_offset = addr_mask_scalar;

        uint32_t pp_n_scalar = block_size; // 64
        uint32_t sub_n_loop = pp_n_scalar / block_size;
        uint32_t real_n_loop = (cur_kv_seqlen + block_size - 1) / block_size;

        uint32_t n_loop = (cur_kv_seqlen + pp_n_scalar - 1) / pp_n_scalar;

        uint32_t qk_n = pp_n_scalar;
        uint32_t qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);

        uint32_t qk_n_2 = pp_n_scalar;
        uint32_t qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);

        // split head num to two vectors
        uint32_t sub_head_num = (sub_block_idx == 1) ? (cur_head_num - cur_head_num / 2) : cur_head_num / 2; // 16
        uint32_t sub_m = sub_head_num * cur_q_seqlen; // 16 * 3 = 48

        uint32_t head_idx = (sub_block_idx == 0) ? start_head : start_head + cur_head_num / 2 * cur_q_seqlen; // not used

        o_offset = addr_o_scalar + start_head * embedding_size + sub_block_idx * cur_head_num / 2 * embedding_size; // for NSD -> SND

        uint32_t sub_m_d128 = (sub_m + 127) / 128;  // up aligned to 128
        uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 128
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;

        uint32_t start_kv = 0;
        uint32_t s_block_stack = 4;
        uint32_t m_slice = FLOAT_VECTOR_SIZE / s_block_stack;
        uint32_t m_end = (sub_m + m_slice - 1) / m_slice;
        for (uint32_t n_idx = 0; n_idx < n_loop + s_block_stack; n_idx += s_block_stack) {
            if (n_idx < n_loop) {
                if (n_idx + s_block_stack > n_loop - 1) {
                    qk_n = (cur_kv_seqlen - n_idx * pp_n_scalar);
                } else {
                    qk_n =  pp_n_scalar * s_block_stack;
                }
                qk_round_n = RoundUp<16>(qk_n);
                if (sub_m == 0) {
                    WaitFlagDev(QK_READY_DECODER);
                }
                uint32_t pingpong_flag = 0;
                for (uint32_t m_ind = 0; m_ind < m_end; m_ind++) {
                    uint32_t row_offset = m_ind * m_slice;
                    uint32_t curr_m = m_ind == m_end - 1 ? sub_m - row_offset : m_slice;
                    uint32_t s_ub_offset = pingpong_flag * 8192;
                    uint32_t p_gm_offset = (uint64_t)block_idx * TMP_SIZE * 2 +
                                            (uint64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n + row_offset * qk_round_n + (uint64_t)((n_idx / s_block_stack) % 2) * TMP_SIZE;
                    uint32_t s_gm_offset = (int64_t)block_idx * TMP_SIZE_DECODER * 4 +
                                            (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n + row_offset * qk_round_n + (uint64_t)((n_idx / s_block_stack) % 2) * TMP_SIZE_DECODER * 2;
                    if (m_ind == 0) {
                        WaitFlagDev(QK_READY_DECODER);
                    }
                    if (curr_m == 0) {
                        continue;
                    }
                    OnlineSoftmaxStage1<float, float, IN_DTYPE, IN_DTYPE, MaskType::MASK_TYPE_NONE> (
                        ls32_ubuf_tensor[s_ub_offset],
                        mask_ubuf_tensor,
                        mask_ubuf_tensor.template ReinterpretCast<float>(),
                        lm32_ubuf_tensor[row_offset],
                        hm32_ubuf_tensor[row_offset],
                        gm32_ubuf_tensor[row_offset],
                        dm32_ubuf_tensor[((n_idx / s_block_stack) % 2) * UB_FLOAT_LINE_SIZE + row_offset],
                        ls32_ubuf_tensor[s_ub_offset],
                        ll_ubuf_tensor[row_offset],
                        gl32_ubuf_tensor[row_offset],
                        lp_ubuf_tensor[s_ub_offset * 2],
                        tv32_ubuf_tensor,
                        s_gm_tensor[s_gm_offset],
                        p_gm_tensor[p_gm_offset],
                        n_idx == 0, this->tor,
                        curr_m, qk_n, qk_round_n, pingpong_flag
                    );
                    pingpong_flag = 1 - pingpong_flag;
                }
                FftsCrossCoreSync<PIPE_MTE3, 2>(SOFTMAX_READY_DECODER);
            }
            /* ************ softmax2 stage1  ************* */
            // PIPE_BARRIER(ALL);
            uint32_t process_row_num = 16;
            uint32_t numhead_per_process = process_row_num / cur_q_seqlen;

            if (n_idx >= s_block_stack) {
                if (n_idx == n_loop) {
                    qk_n_2 = (cur_kv_seqlen - (n_idx - 1) * pp_n_scalar);
                    qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);
                }
                WaitFlagDev(UPDATE_READY_DECODER);
                if (sub_m > 0) {
                    uint32_t head_loop = (sub_m + process_row_num - 1) / process_row_num;

                    uint32_t head_res_row_num = 0;
                    uint32_t head_start_sblock_idx = 0;
                    uint32_t tail_res_row_num = 0;

                    for (uint32_t head_loop_idx = 0; head_loop_idx < head_loop; ++head_loop_idx) {
                        uint32_t head_offset = head_loop_idx * process_row_num * round_v;
                        uint32_t cur_sub_m = head_loop_idx == (head_loop - 1) ? sub_m - head_loop_idx * process_row_num : process_row_num;

                        // complete head num
                        head_start_sblock_idx = tail_res_row_num;
                        head_res_row_num = (cur_q_seqlen - tail_res_row_num) % cur_q_seqlen;
                        uint32_t cur_numhead_per_process = (cur_sub_m - head_res_row_num) / cur_q_seqlen;
                        tail_res_row_num = cur_sub_m - cur_numhead_per_process * cur_q_seqlen - head_res_row_num;

                        uint32_t out_o_offset = head_loop_idx * numhead_per_process * round_v; // round_v = 512

                        TailSoftmaxStage2MLAHeadLoopTP1(
                            o_tmp_gm_tensor[(uint64_t)(block_idx * TMP_SIZE * 2 + sub_block_idx * cur_head_num * cur_q_seqlen / 2 * round_v + head_offset + ((n_idx / s_block_stack - 1) % 2) * TMP_SIZE)],
                            go_gm_tensor[(uint64_t)(block_idx * TMP_SIZE + sub_block_idx * cur_head_num * cur_q_seqlen / 2 * round_v + head_offset)],
                            tmp_gm_tensor[(uint64_t)(block_idx * q_heads + start_head + sub_block_idx * cur_head_num * cur_q_seqlen / 2 + head_loop_idx * process_row_num)],
                            tmp_gm_tensor[(uint64_t)(block_num * q_heads + block_idx * q_heads + start_head + sub_block_idx * cur_head_num * cur_q_seqlen / 2 + head_loop_idx * process_row_num)],
                            dm32_ubuf_tensor[(uint64_t)((n_idx / s_block_stack - 1) % 2 * UB_FLOAT_LINE_SIZE + head_loop_idx * process_row_num)],
                            go32_ubuf_tensor, // no need for offset
                            gl32_ubuf_tensor[(uint64_t)(head_loop_idx * process_row_num)],
                            gm32_ubuf_tensor[(uint64_t)(head_loop_idx * process_row_num)],
                            n_idx, n_loop, qk_n_2, RoundUp<T_BLOCK_SIZE>(qk_round_n_2), cur_sub_m, o_offset, head_idx,
                            head_loop, head_loop_idx, cur_q_seqlen, sub_head_num, cur_head_num, cur_numhead_per_process
                        );
                    }
                }
            }
        }

    }

    __aicore__ __attribute__((always_inline)) inline void TailInnerGatherVectorTP1(
        uint32_t start_head, uint32_t cur_q_seqlen, uint32_t cur_head_num,
        uint32_t start_block_idx, uint32_t cores_process, uint32_t offset_tiling)
    {
        uint32_t prev_task = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 1 + offset_tiling));
        uint64_t addr_o_scalar = prev_task * q_heads * embedding_size;
        // split head num to two vectors
        uint32_t sub_head_num = (sub_block_idx == 1) ? (cur_head_num - cur_head_num / 2) : cur_head_num / 2; // 16
        uint32_t sub_m = sub_head_num * cur_q_seqlen;
        uint32_t head_idx = (sub_block_idx == 0) ? start_head : start_head + cur_head_num / 2 * cur_q_seqlen; // not used
        
        uint32_t process_row_num = 16;
        uint32_t numhead_per_process = process_row_num / cur_q_seqlen;
        uint32_t head_loop = (sub_m + process_row_num - 1) / process_row_num;

        
        o_offset = addr_o_scalar + start_head * embedding_size + sub_block_idx * cur_head_num / 2 * embedding_size;
        
        for (uint32_t head_loop_idx = 0; head_loop_idx < head_loop; ++head_loop_idx) {
            uint32_t head_offset = head_loop_idx * process_row_num;
            uint32_t cur_sub_m = head_loop_idx == (head_loop - 1) ? sub_m - head_loop_idx * process_row_num : process_row_num;
            uint32_t out_o_offset = head_loop_idx * numhead_per_process * embedding_size;

            for (uint32_t core_idx = 0; core_idx < cores_process; core_idx++) {
                uint32_t actual_block_idx = start_block_idx + core_idx;

                SoftmaxGatherTP1(
                    o_gm_tensor[o_offset + out_o_offset],
                    go_gm_tensor[(uint64_t)(actual_block_idx * TMP_SIZE + start_head * embedding_size + sub_block_idx * cur_head_num * cur_q_seqlen / 2 * round_v + head_offset * embedding_size)],
                    tmp_gm_tensor[(uint64_t)(actual_block_idx * q_heads + start_head + sub_block_idx * cur_head_num * cur_q_seqlen / 2 + head_offset)],
                    tmp_gm_tensor[(uint64_t)(block_num * q_heads + actual_block_idx * q_heads + start_head + sub_block_idx * cur_head_num * cur_q_seqlen / 2 + head_offset)],
                    go_ubuf_tensor,
                    go32_ubuf_tensor,
                    gl32_ubuf_tensor,
                    gm32_ubuf_tensor,
                    lo_ubuf_tensor,
                    ll_ubuf_tensor,
                    lm32_ubuf_tensor,
                    hm32_ubuf_tensor,
                    dm32_ubuf_tensor,
                    tv32_ubuf_tensor,
                    start_block_idx, cores_process, actual_block_idx, cur_sub_m
                );
            }
            
        }
    }

private:

    __gm__ mm1CopyType *__restrict__ s_gm{nullptr};
    __gm__ IN_DTYPE *__restrict__ p_gm{nullptr};
    __gm__ mm2CopyType *__restrict__ o_tmp_gm{nullptr};
    __gm__ float *__restrict__ go_gm{nullptr};
    __gm__ int32_t* __restrict__ gm_block_tables_{nullptr};
    __gm__ OUT_DTYPE *__restrict__ lse_gm{nullptr};
    __gm__ OUT_DTYPE *__restrict__ o_gm{nullptr};
    __gm__ OUT_DTYPE *__restrict__ mask_gm{nullptr};
    __gm__ uint8_t *__restrict__ tiling_gm{nullptr};

    UbufAlloc<blockStack> UbAllocator;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> ls32_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.ls32_ubuf_offset);
    AscendC::LocalTensor<float> ls32_quant_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.ls32_quant_ubuf_offset);
    AscendC::LocalTensor<half> ls16_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(UbAllocator.ls32_ubuf_offset);
    AscendC::LocalTensor<IN_DTYPE> lp_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, IN_DTYPE>(UbAllocator.lp_ubuf_offset);
    AscendC::LocalTensor<float> lp32_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.lp32_ubuf_offset);
    AscendC::LocalTensor<OUT_DTYPE> mask_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(UbAllocator.mask_ubuf_offset);
    AscendC::LocalTensor<float> lo_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.lo_ubuf_offset);
    AscendC::LocalTensor<float> mask32_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.mask32_ubuf_offset);
    AscendC::LocalTensor<float> lm32_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.lm32_ubuf_offset);
    AscendC::LocalTensor<float> hm32_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.hm32_ubuf_offset);
    AscendC::LocalTensor<float> pm32_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.pm32_ubuf_offset);
    AscendC::LocalTensor<float> pm32_ubuf_stage2_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.pm32_ubuf_stage2_offset);
    AscendC::LocalTensor<float> gm32_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.gm32_ubuf_offset);
    AscendC::LocalTensor<float> dm32_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.dm32_ubuf_offset);
    AscendC::LocalTensor<float> descale_q1_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.descale1_offset);
    AscendC::LocalTensor<float> descale_k1_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.descale2_offset);

    AscendC::LocalTensor<OUT_DTYPE> lse_conv_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(UbAllocator.tv32_ubuf_offset);
    AscendC::LocalTensor<float> lse32_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.tv32_ubuf_offset);

    AscendC::LocalTensor<float> dm32_stage2_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.dm32_ubuf_stage2_offset);
    AscendC::LocalTensor<float> ll_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.ll_ubuf_offset);
    AscendC::LocalTensor<float> ll_stage2_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.ll_ubuf_stage2_offset);
    AscendC::LocalTensor<OUT_DTYPE> gl_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(UbAllocator.gl_ubuf_offset);
    AscendC::LocalTensor<float> gl32_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.gl32_ubuf_offset);
    AscendC::LocalTensor<float> tv32_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.tv32_ubuf_offset);
    AscendC::LocalTensor<OUT_DTYPE> go_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(UbAllocator.go_ubuf_offset);
    AscendC::LocalTensor<float> go32_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.go32_ubuf_offset);


    AscendC::GlobalTensor<OUT_DTYPE> mask_gm_tensor;
    AscendC::GlobalTensor<OUT_DTYPE> o_gm_tensor;
    AscendC::GlobalTensor<OUT_DTYPE> lse_gm_tensor;
    AscendC::GlobalTensor<mm1CopyType> s_gm_tensor;
    AscendC::GlobalTensor<float> s_rope_gm_tensor;
    AscendC::GlobalTensor<IN_DTYPE> p_gm_tensor;
    AscendC::GlobalTensor<mm2OutputType> o_tmp_gm_tensor;
    AscendC::GlobalTensor<float> go_gm_tensor;
    AscendC::GlobalTensor<float> tmp_gm_tensor;
    AscendC::GlobalTensor<float> deq_scale_gm_tensor_q1;
    AscendC::GlobalTensor<float> deq_scale_gm_tensor_k1;

    uint32_t go_flag_scalar{1};
    uint32_t gl_flag_scalar{1};
    uint32_t pm_flag_scalar1{1};
    uint32_t pm_flag_scalar2{0};
    uint32_t num_batches{0};
    uint32_t q_heads{0};
    uint32_t num_kv_heads{0};
    uint32_t embedding_size{0};
    uint32_t block_size{0};
    uint32_t max_context_len{0};
    uint32_t start_head{0};
    uint32_t cur_head_num{0};
    uint32_t __k{0};
    uint32_t round_k{0};
    uint32_t __v{0};
    uint32_t round_v{0};
    uint32_t cur_batch{0};
    float tor{0};
    uint64_t sub_block_idx{0};
    uint32_t batch_stride{0};
    uint32_t core_per_batch{0};
    uint32_t process_num{0};
    uint32_t tiling_head_size{0};
    uint32_t tiling_para_size{0};
    uint32_t block_size_calc{0};
    uint32_t mask_type{0};
    uint32_t embed_split_size_v_former{0};
    uint32_t embed_split_loop_v_former{1};
    uint32_t embed_split_size_v_tail{0};
    uint32_t embed_split_loop_v_tail{1};
    uint32_t max_num_blocks_per_query{0};
    uint64_t o_offset{0};
    uint32_t totalTaskNum{0};
    uint32_t maxKVSeqLen{0};

    uint32_t cur_qn_blk_size{0};
};
#endif
