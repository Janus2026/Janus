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

#include <array>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <securec.h>
#include "exe_graph/runtime/tiling_context.h"
#include "multi_latent_attention_tiling_impl.h"

namespace AtbOps {
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
const int32_t TILING_TASK_NUM = 13;
const int32_t TILING_MAX_KV_SEQ_LEN = 14;

const int32_t NUM0 = 0;
const int32_t NUM1 = 1;
const int32_t NUM2 = 2;
const int32_t NUM3 = 3;
const int32_t NUM4 = 4;
const int32_t NUM5 = 5;
const int32_t NUM6 = 6;
const int32_t NUM7 = 7;
const int32_t NUM8 = 8;
const int32_t NUM9 = 9;
const int32_t NUM10 = 10;
const int32_t NUM11 = 11;
const int32_t NUM12 = 12;
const int32_t NUM13 = 13;
const int32_t NUM14 = 14;
const int32_t NUM15 = 15;
const int32_t NUM16 = 16;
const int32_t NUM17 = 17;
const int32_t NUM18 = 18;
const int32_t NUM19 = 19;
const int32_t NUM20 = 20;
const int32_t NUM21 = 21;
const int32_t NUM32 = 32;
const int32_t NUM64 = 64;
const int32_t NUM128 = 128;
const int32_t NUM256 = 256;
const int32_t NUM512 = 512;
const int32_t NUM576 = 576;
const int32_t INDEX125 = 125;
const int32_t INDEX126 = 126;
const int32_t INDEX127 = 127;
const int32_t INDEX190 = 190;
const int32_t INDEX191 = 191;
const int32_t SPECIALNUM_TOKENS = 16;
const int32_t SPECIALNUM_HEADS = 32;
const int32_t EMBEDDING_LIMIT = 128;
const int32_t HIGH_32BIT = 32;
const uint32_t BATCH_MLA = 32;
const uint32_t BLOCK_DIM_MLA = 20;
const int32_t PP_MM_NUM = 8;
const int32_t PP_INDEX = 16;

constexpr std::array<int32_t, PP_MM_NUM> PP_MM = { 16, 32, 48, 64, 80, 96, 112, 128 };
constexpr std::array<int32_t, NUM6> QN_TILE_LIST = { 128, 64, 32, 16, 8, 1 };

using IndexArr = std::array<int32_t, NUM4>;

inline uint32_t GetHigh32Bit(uint64_t v) { return static_cast<uint32_t>(v >> HIGH_32BIT); }
inline uint32_t GetLoww32Bit(uint64_t v) { return static_cast<uint32_t>(v); }

inline int32_t ConvertValueToIndexMM(int32_t val, int32_t idxBound) // 16, 7
{
    return (val > PP_MM[idxBound]) ? idxBound : (val / PP_INDEX - 1);
}

void GetAddrOffsetMLA(uint32_t *tilingParam, const AddrOffsets addrOffsets, const int32_t tilingOffset)
{
    tilingParam[tilingOffset + NUM2] = GetHigh32Bit(addrOffsets.addrQSeqOffset);
    tilingParam[tilingOffset + NUM3] = GetLoww32Bit(addrOffsets.addrQSeqOffset);
    tilingParam[tilingOffset + NUM4] = GetHigh32Bit(addrOffsets.addrOSeqOffset);
    tilingParam[tilingOffset + NUM5] = GetLoww32Bit(addrOffsets.addrOSeqOffset);

    // mask offset
    tilingParam[tilingOffset + NUM6] = GetHigh32Bit(addrOffsets.addrMaskOffset);
    tilingParam[tilingOffset + NUM7] = GetLoww32Bit(addrOffsets.addrMaskOffset);
}

int32_t GetQNBlockTile(const MLAInfo &mmInfo, int32_t qSeqLen)
{
    int32_t tileListIdx = static_cast<int32_t>(std::ceil(std::log2(qSeqLen)));
    tileListIdx = (tileListIdx > NUM5) ? NUM5 : tileListIdx;
    int32_t qNBlockTile = QN_TILE_LIST[tileListIdx];
    int32_t group = mmInfo.numHeads / mmInfo.kvHeads;
    qNBlockTile = (qNBlockTile > group) ? group : qNBlockTile;

    return qNBlockTile;
}

int32_t GetMaxQseqlen(const OpParam::MLA &param)
{
    auto qSeqLen = param.qSeqLen;
    auto maxQSeqlenIter = std::max_element(qSeqLen.begin(), qSeqLen.end());
    auto maxQseqlen = maxQSeqlenIter != qSeqLen.end() ? *maxQSeqlenIter : 1;
    return maxQseqlen;
}

int32_t GetMaxKVseqlen(const OpParam::MLA &param)
{
    auto kvSeqLen = param.kvSeqLen;
    auto maxKVSeqlenIter = std::max_element(kvSeqLen.begin(), kvSeqLen.end());
    auto maxKVseqlen = maxKVSeqlenIter != kvSeqLen.end() ? *maxKVSeqlenIter : 1;
    return maxKVseqlen;
}

ge::graphStatus GetNdMLATiling(const MLAInfo &mmInfo, uint32_t &blockDim, uint32_t *tilingParam,
                      const OpParam::MLA &param)
{
    AddrOffsets addrOffsets {};

    auto qSeqLen = param.qSeqLen;
    int32_t maxQseqlen =  GetMaxQseqlen(param);
    if (maxQseqlen <= 0) {
        printf("qSeqlen max value(%d) invalid, please check\n", maxQseqlen);
        return ge::GRAPH_FAILED;
    }

    int32_t maxKVseqlen =  GetMaxKVseqlen(param);
    if (maxKVseqlen <= 0) {
        printf("kvSeqlen max value(%d) invalid, please check\n", maxKVseqlen);
        return ge::GRAPH_FAILED;
    }

    int32_t curQNBlockTile = GetQNBlockTile(mmInfo, maxQseqlen);

    uint32_t emptySeq = (mmInfo.qSeqLen == nullptr) ? 1 : 0;
    for (int32_t seqIdx = 0; seqIdx < mmInfo.batch; seqIdx++) {
        int32_t qSeqLen = 1;
        qSeqLen = (emptySeq == 1) ? 1 : *(mmInfo.qSeqLen + seqIdx);
        qSeqLen = (*(mmInfo.kvSeqLen + seqIdx) == 0) ? 0 : qSeqLen;
        int32_t kvSeqlen = *(mmInfo.kvSeqLen + seqIdx);

        int32_t tilingOffset = TILING_HEAD_SIZE + TILING_PARA_SIZE * seqIdx;
        tilingParam[tilingOffset] = static_cast<uint32_t>(qSeqLen);
        tilingParam[tilingOffset + 1] = static_cast<uint32_t>(kvSeqlen);

        GetAddrOffsetMLA(tilingParam, addrOffsets, tilingOffset);
        uint64_t addressQffset = static_cast<uint64_t>(mmInfo.numHeads * qSeqLen);
        uint64_t addressOffset = static_cast<uint64_t>(mmInfo.numHeads * mmInfo.embeddingSize * qSeqLen);
        uint64_t addressMaskOffset = static_cast<uint64_t>(qSeqLen * maxKVseqlen);
        addrOffsets.addrQSeqOffset += addressQffset;
        addrOffsets.addrOSeqOffset += addressOffset;
        addrOffsets.addrMaskOffset += addressMaskOffset;
    }

    tilingParam[TILING_MTP_HEAD_SPLIT_SIZE] = static_cast<uint32_t>(curQNBlockTile);
    tilingParam[TILING_MAX_KV_SEQ_LEN] = static_cast<uint32_t>(maxKVseqlen);
    return ge::GRAPH_SUCCESS;
}

void GetNdMLAMtpTilingTP1(const MLAInfo &mmInfo, uint32_t &blockDim, uint32_t *tilingParam,
                          const OpParam::MLA &param)
{
    int32_t prevTaskNum = 0;
    for (int32_t seqIdx = 0; seqIdx < mmInfo.batch; seqIdx++) {
        int32_t qSeqLen = mmInfo.qSeqLen == nullptr ? 1 : *(mmInfo.qSeqLen + seqIdx);
        int32_t kvSeqlen = *(mmInfo.kvSeqLen + seqIdx);
        for (int32_t qSeq = 0; qSeq < qSeqLen; qSeq++) {
            int32_t tilingOffset = TILING_HEAD_SIZE + TILING_PARA_SIZE_TP1 * prevTaskNum;
            tilingParam[tilingOffset] = seqIdx;
            tilingParam[tilingOffset + NUM1] = prevTaskNum;
            tilingParam[tilingOffset + NUM2] = kvSeqlen - qSeqLen + qSeq + 1;
            prevTaskNum++;
        }
    }
}

void GetTilingHead(const MLAInfo &mmInfo, const OpParam::MLA &param, uint32_t *tilingParam,
                   const uint32_t *torPtr)
{
    tilingParam[TILING_BATCH] = static_cast<uint32_t>(mmInfo.batch);
    tilingParam[TILING_HEADSIZE] = static_cast<uint32_t>(TILING_HEAD_SIZE);
    tilingParam[TILING_PARASIZE] = mmInfo.mtpTp1Flag ? static_cast<uint32_t>(TILING_PARA_SIZE_TP1) :
                                    static_cast<uint32_t>(TILING_PARA_SIZE);
    tilingParam[TILING_NUMHEADS] = static_cast<uint32_t>(mmInfo.numHeads);
    tilingParam[TILING_HEADDIM] = static_cast<uint32_t>(mmInfo.embeddingSize);
    tilingParam[TILING_NUMBLOKS] = static_cast<uint32_t>(mmInfo.numBlocks);
    tilingParam[TILING_BLOCKSIZE] = static_cast<uint32_t>(mmInfo.blockSize);
    tilingParam[TILING_MAXBLOCKS] = static_cast<uint32_t>(mmInfo.maxNumBlocksPerQuery);
    tilingParam[TILING_TOR] = *torPtr;
    tilingParam[TILING_KVHEADS] = (mmInfo.kvHeads == 0) ? mmInfo.numHeads : mmInfo.kvHeads;

    tilingParam[TILING_MASK_TYPE_ND] = static_cast<uint32_t>(mmInfo.maskType);
    tilingParam[TILING_TASK_NUM] = static_cast<uint32_t>(mmInfo.totalTaskNum);

}

ge::graphStatus GetMLATilingParam(OpParam::MLA param, const MLAInfo &mmInfo,
    uint32_t &blockDim, uint32_t *tilingParam, uint64_t tilingParamSize)
{
    float tor = mmInfo.tor;
    uint32_t *torPtr = reinterpret_cast<uint32_t *>(&tor);

    uint64_t curTilingParamSize = mmInfo.mtpTp1Flag ?
                                  (TILING_HEAD_SIZE + TILING_PARA_SIZE_TP1 * mmInfo.totalTaskNum) * sizeof(uint32_t) :
                                  (TILING_HEAD_SIZE + TILING_PARA_SIZE * mmInfo.batch) * sizeof(uint32_t);

    if (mmInfo.mtpTp1Flag) {
        GetNdMLAMtpTilingTP1(mmInfo, blockDim, tilingParam, param);
    } else {
        ge::graphStatus ret = GetNdMLATiling(mmInfo, blockDim, tilingParam, param);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }

        blockDim = mmInfo.batch == BATCH_MLA ? BLOCK_DIM_MLA : blockDim;
    }
    GetTilingHead(mmInfo, param, tilingParam, torPtr);
    return ge::GRAPH_SUCCESS;
}
}
