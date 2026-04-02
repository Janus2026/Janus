/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef X_FLASH_ATTENTION_INFER_COMMON
#define X_FLASH_ATTENTION_INFER_COMMON

struct CoreNode {
    uint32_t startBIdx{0};
    uint32_t startN1Idx{0};
    uint32_t startS2Idx{0};
    uint32_t endBIdx{0};
    uint32_t endN1Idx{0};
    uint32_t endS2Idx{0};
    uint64_t firstSplitKVTaskLseOffset{0};
    uint64_t firstSplitKVTaskOOffset{0};
};

struct SplitNode {
    uint32_t batchIdx{0};
    uint32_t headStartIdx{0};
    uint32_t headEndIdx{0};
    uint32_t qStartIdx{0};
    uint32_t qEndIdx{0};
    uint32_t splitNum{0};
    uint64_t lseTaskOffset{0};
    uint64_t oTaskOffset{0};
};

struct SplitKvExtraInfo {
    CoreNode coreInfo[25];
    SplitNode splitInfo[25];
    uint32_t totalSplitNodeNum;
};

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

constexpr uint32_t QK_READY_ID = 1;
constexpr uint32_t SOFTMAX_READY_ID = 2;
constexpr uint32_t PV_READY_ID = 3;
constexpr uint32_t PRE_LAUNCH = 2;
constexpr uint32_t N_SPLIT_HELPER = 2;
constexpr uint32_t MAX_KV_STACK_LEN = 1024;
constexpr uint32_t Q_TILE_CEIL = 128;
constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
constexpr uint32_t L1_MAX_SIZE = 524288;
constexpr uint32_t L1_MAX_N_NUM = 128;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t COMP_TRIU_MASK_DIM_LEN = 2048;
constexpr uint32_t NUM_32 = 32;
constexpr uint32_t NUM_128 = 128;
constexpr uint32_t NUM_256 = 256;

/**
 * flash_attention_infer TilingKey 定义
 *
 * TilingKey编码规则 (64-bit):
 * - [0]  Mask Type: 0=NoMask, 3=CausalMask
 * - [1]  qkv dtype: 1=FP16, 2=BF16
 * - [2]  KV Layout: 1=TND, 2=NZ
 * - [3]  FD: 0=disable, 1=enable
 */

#define FAI_BASE_TILING 1000000000000000000

#define QFP16_KVFP16_TND_CAUSALMASK_NOFD_TILING 1000000000000000113
#define QFP16_KVFP16_KVNZ_CAUSALMASK_NOFD_TILING  1000000000000000213
#define QBF16_KVBF16_TND_CAUSALMASK_NOFD_TILING 1000000000000000123
#define QBF16_KVBF16_KVNZ_CAUSALMASK_NOFD_TILING  1000000000000000223

#define QFP16_KVFP16_TND_CAUSALMASK_FD_TILING   1000000000000001113
#define QFP16_KVFP16_KVNZ_CAUSALMASK_FD_TILING    1000000000000001213
#define QBF16_KVBF16_TND_CAUSALMASK_FD_TILING   1000000000000001123
#define QBF16_KVBF16_KVNZ_CAUSALMASK_FD_TILING    1000000000000001223

template <typename T>
    __aicore__ inline
    T AlignUp(T a, T b)
    {
        return (b == 0) ? 0 : (a + b - 1) / b * b;
    }

template <typename T>
    __aicore__ inline
    T Max(T a, T b)
    {
        return (a > b) ? a : b;
    }

namespace FaiKenel {
    constexpr uint32_t BLOCK_SIZE = 16;

    enum class cvPipeLineType : uint32_t {
        FAI_COMMON_NORMAL = 0,
        FAI_COMMON_CHUNK_MASK = 1,
    };

    enum class MaskType : uint32_t {
        NO_MASK = 0,
        MASK_CAUSAL = 1,
        MASK_SPEC = 2
    };

    enum class inputLayout : uint32_t {
        BSND = 0,
        TND = 1
    };
};


__aicore__ inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qNBlockTile = (qSeqlen != 0) ?
        (Q_TILE_CEIL / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
    qNBlockTile = qNBlockTile < groupSize ? qNBlockTile : groupSize;
    qNBlockTile = qNBlockTile < 1 ? 1 : qNBlockTile;
    return qNBlockTile;
}

__aicore__ inline uint32_t GetQSBlockTile(uint32_t kvSeqlen)
{
    uint32_t qSBlockTile = Q_TILE_CEIL;
    return qSBlockTile;
}

__aicore__ inline uint32_t GetKSBlockTile(uint32_t kvSeqlen)
{
    uint32_t kSBlockTile = MAX_KV_STACK_LEN;
    return kSBlockTile;
}

struct FAIKernelParams {
    // Data members
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR blockTables;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    GM_ADDR o;
    GM_ADDR s;
    GM_ADDR p;
    GM_ADDR oTemp;
    GM_ADDR oUpdate;
    GM_ADDR tiling;

    // Methods
    __aicore__ inline FAIKernelParams() {}

    __aicore__ inline FAIKernelParams(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_,
            GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR o_, GM_ADDR s_, GM_ADDR p_, GM_ADDR oTemp_, GM_ADDR oUpdate_, GM_ADDR tiling_)
        : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
            actualKvseqlen(actualKvseqlen_), o(o_), s(s_), p(p_), oTemp(oTemp_), oUpdate(oUpdate_), tiling(tiling_) {}

};

struct FAIFDKernelParams {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR blockTables;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    GM_ADDR o;
    GM_ADDR s;
    GM_ADDR p;
    GM_ADDR oTemp;
    GM_ADDR oUpdate;
    GM_ADDR gmlse;
    GM_ADDR glo;
    GM_ADDR tiling;
    __gm__ SplitKvExtraInfo *extraInfo;

    // Methods
    __aicore__ inline FAIFDKernelParams() {}


    __aicore__ inline FAIFDKernelParams(GM_ADDR q_,
                                        GM_ADDR k_,
                                        GM_ADDR v_,
                                        GM_ADDR mask_,
                                        GM_ADDR blockTables_,
                                        GM_ADDR actualQseqlen_,
                                        GM_ADDR actualKvseqlen_,
                                        GM_ADDR o_,
                                        GM_ADDR s_,
                                        GM_ADDR p_,
                                        GM_ADDR oTemp_,
                                        GM_ADDR oUpdate_,
                                        GM_ADDR gmlse_,
                                        GM_ADDR glo_,
                                        GM_ADDR tiling_,
                                        __gm__ SplitKvExtraInfo *extraInfo_)
                            : q(q_)
                            , k(k_)
                            , v(v_)
                            , mask(mask_)
                            , blockTables(blockTables_)
                            , actualQseqlen(actualQseqlen_)
                            , actualKvseqlen(actualKvseqlen_)
                            , o(o_)
                            , s(s_)
                            , p(p_)
                            , oTemp(oTemp_)
                            , oUpdate(oUpdate_)
                            , gmlse(gmlse_)
                            , glo(glo_)
                            , tiling(tiling_)
                            , extraInfo(extraInfo_) {
                        }
};
#endif