/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_COMMON
#define KERNEL_COMMON

constexpr uint32_t QK_READY_ID = 1;
constexpr uint32_t SOFTMAX_READY_ID = 2;
constexpr uint32_t PV_READY_ID = 3;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr int32_t WORKSPACE_BLOCK_SIZE_DB = 128 * 128 * 4; // row * col * blockStackNum
constexpr int32_t UNSHARED_WORKSPACE_BLOCK_SIZE_DB = 128 * 256;   // unshared no pinpong
constexpr uint32_t TMP_SIZE_DECODER = 32768;
constexpr uint32_t PRE_LAUNCH = 2;
constexpr uint32_t SOFTMAX_BROAD_SIZE = 8;

constexpr int32_t TILING_BATCH = 0;
constexpr int32_t TILING_NUMHEADS = 1;
constexpr int32_t TILING_HEADDIM = 2;
constexpr int32_t TILING_NUMBLOKS = 3;
constexpr int32_t TILING_BLOCKSIZE = 4;
constexpr int32_t TILING_MAXBLOCKS = 5;
constexpr int32_t TILING_TOR = 6;
constexpr int32_t TILING_KVHEADS = 7;
constexpr int32_t TILING_HEADSIZE = 8;
constexpr int32_t TILING_PARASIZE = 9;
constexpr int32_t TILING_HEAD_SPLIT_SIZE = 10;
constexpr int32_t TILING_HEAD_SPLIT_NUM = 11;
constexpr int32_t TILING_HEADDIM_ROPE = 13;
constexpr int32_t TILING_MAX_KVSEQLEN = 14;
constexpr int32_t TILING_KVSPLIT = 15;
constexpr int32_t TILING_KVCORENUM = 16;
constexpr int32_t TILING_TOTAL_QTOKENS = 18;
constexpr int32_t TILING_FORMERTASKNUM = 19;
constexpr int32_t TILING_TAILTASKNUM = 20;
constexpr int32_t TILING_BLOCKSIZE_CALC = 25;
constexpr int32_t TILING_HEADDIM_K_SPLIT = 38;
constexpr int32_t TILING_HEADDIM_V_SPLIT = 39;
constexpr int32_t TILING_HEADDIM_V_SPLIT_VECTOR_FORMER = 40;
constexpr int32_t TILING_HEADDIM_V_SPLIT_VECTOR_TAIL = 41;

constexpr int32_t NUM1 = 1;
constexpr int32_t NUM2 = 2;
constexpr int32_t NUM3 = 3;
constexpr int32_t NUM4 = 4;
constexpr int32_t NUM64 = 64;
constexpr int32_t NUM512 = 512;
constexpr int32_t NUM576 = 576;

constexpr uint32_t FLOAT_VECTOR_SIZE = 64;

constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;

template <typename T>
CATLASS_DEVICE T AlignUp(T a, T b) {
    return (b == 0) ? 0 : (a + b - 1) / b * b;
}

template <typename T>
CATLASS_DEVICE T Min(T a, T b) {
    return (a > b) ? b : a;
}

template <typename T>
CATLASS_DEVICE T Max(T a, T b) {
    return (a > b) ? a : b;
}

enum class cvPipeLineType {
    FAI_COMMON_NORMAL = 0,
    FAI_COMMON_CHUNK_MASK = 1
};

CATLASS_DEVICE
uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize) {
    uint32_t qNBlockTile = (128 / qSeqlen) / 2 * 2;
    qNBlockTile = qNBlockTile < groupSize ? qNBlockTile : groupSize;
    qNBlockTile = qNBlockTile < 1 ? 1 : qNBlockTile;
    return qNBlockTile;
}

CATLASS_DEVICE
uint32_t GetQSBlockTile(uint32_t kvSeqlen) {
    uint32_t qSBlockTile = 128;
    return qSBlockTile;
}

struct XATilingData {
    // common TilingData
    uint32_t numHeads = 0;
    uint32_t kvHeads = 0;
    uint32_t embeddingSize = 0;
    uint32_t batch = 0;  // request num
    uint32_t beamSize = 0;
    float scaleValue = 0;
    uint32_t maskType = 0;
    uint32_t numTokens = 0; // batch * beamSize
    // Shared TilingData
    uint32_t numBlocks = 0;
    uint32_t blockSize = 0;
    uint32_t sharedCoreNum = 0;
    uint32_t maxKvSeqlen = 0;
    uint32_t maxNumBlocksPerBatch = 0;
    uint32_t firstSharedBatchTaskNum = 0;
    uint32_t sharedTotalTaskNum = 0;
    uint64_t mm1OutSize = 0;
    uint64_t smOnlineOutSize = 0;
    uint64_t mm2OutSize = 0;
    uint64_t updateSize = 0;
    uint64_t sharedWorkspaceSize = 0; 
    // shared_gl_and_gm_size: num_tokens * numHeads * [outputAxisSize] * 2(gl and gm) * 2(Bytes)
    // UnSharedTilingData
    uint32_t groupSize = 0;
    uint32_t maxDecodeStep = 0;
    uint32_t unsharedCoreNum = 0;
    uint32_t unshareGroupCountPerLoop = 0;
    uint32_t unshareGroupCountTailLoop = 0;
    uint32_t unsharedFullCoreNum = 0;
    uint32_t unsharedTaskNumHead = 0;
    uint32_t unsharedTaskNumTail = 0;

    // CombineTilingData
    uint32_t combineFormerCoreNum = 0;
    uint32_t combineFormerRowNum = 0;
    uint32_t combineTailRowNum = 0;
    uint32_t combineCoreNum = 0;
};


struct XAttnKernelParams {
    GM_ADDR q;
    GM_ADDR k_cache;
    GM_ADDR v_cache;
    GM_ADDR unshared_k;
    GM_ADDR unshared_v;
    GM_ADDR sharedBlockTable;
    GM_ADDR unsharedBlockTable;
    GM_ADDR actualKvseqlen; // shared Kv
    GM_ADDR decodeStep;     // unshared kv: 1, 2, 3
    GM_ADDR s;
    GM_ADDR p;
    GM_ADDR oTemp;
    GM_ADDR oUpdate;
    GM_ADDR shared_workspace; // shared_gl and shared_gm
    GM_ADDR unshared_workspace; // unshared_output, unshared_gm and unshare_gl
    GM_ADDR o;  // final combine out
    GM_ADDR tiling;

    CATLASS_DEVICE
    XAttnKernelParams() {
    }

    CATLASS_DEVICE
    XAttnKernelParams(GM_ADDR q_,
    GM_ADDR k_cache_,
    GM_ADDR v_cache_,
    GM_ADDR unshared_k_,
    GM_ADDR unshared_v_,
    GM_ADDR sharedBlockTable_,
    GM_ADDR unsharedBlockTable_,
    GM_ADDR actualKvseqlen_,
    GM_ADDR decodeStep_,
    GM_ADDR s_,
    GM_ADDR p_,
    GM_ADDR oTemp_,
    GM_ADDR oUpdate_,
    GM_ADDR shared_workspace_,
    GM_ADDR unshared_workspace_,
    GM_ADDR o_, GM_ADDR tiling_) : q(q_), k_cache(k_cache_), v_cache(v_cache_),
    unshared_k(unshared_k_), unshared_v(unshared_v_), sharedBlockTable(sharedBlockTable_),
    unsharedBlockTable(unsharedBlockTable_), actualKvseqlen(actualKvseqlen_),
    decodeStep(decodeStep_), s(s_), p(p_), oTemp(oTemp_), oUpdate(oUpdate_),
    shared_workspace(shared_workspace_), unshared_workspace(unshared_workspace_),
    o(o_), tiling(tiling_)
    {}
};


#endif