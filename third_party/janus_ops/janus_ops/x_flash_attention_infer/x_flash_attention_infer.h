/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _ASCENDC_X_FLASH_ATTENTION_INFER_H_
#define _ASCENDC_X_FLASH_ATTENTION_INFER_H_
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
#include "x_flash_attention_infer_common.h"

using namespace Catlass;

template <
    class BlockMmadQK,
    class BlockMmadPV,
    class EpilogueOnlineSoftmax,
    class EpilogueRescaleO,
    bool PAGED_CACHE_FLAG,
    FaiKenel::MaskType MASK_TYPE = FaiKenel::MaskType::NO_MASK,
    FaiKenel::inputLayout INPUT_LAYOUT = FaiKenel::inputLayout::BSND>
class FAInferKernel {
public:
    using ArchTag = typename BlockMmadQK::ArchTag;
    using L1TileShape = typename BlockMmadQK::L1TileShape;
    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = typename BlockMmadQK::LayoutA;
    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutK = typename BlockMmadQK::LayoutB;
    using ElementS = typename BlockMmadQK::ElementC;
    using LayoutS = typename BlockMmadQK::LayoutC;

    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutP = typename BlockMmadPV::LayoutA;
    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutV = typename BlockMmadPV::LayoutB;

    using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
    using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;

    using ElementO = typename EpilogueRescaleO::ElementOutput;
    using LayoutO = typename EpilogueRescaleO::LayoutOutput;

    using ElementOTmp = typename EpilogueRescaleO::ElementInput;
    using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;

    using ElementLse = typename EpilogueRescaleO::ElementLse;
    using LayoutLse = typename EpilogueRescaleO::LayoutLse;

    using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
    using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

    static constexpr Epilogue::LseMode LSE_MODE = EpilogueRescaleO::LSE_MODE;

    // Methods
    __aicore__ inline
    FAInferKernel() {}

    __aicore__ inline
    void operator()(FAIKernelParams const &params)
    {
        __gm__ XFAInferTilingData *fATilingData = reinterpret_cast<__gm__ XFAInferTilingData *>(params.tiling);
        uint64_t mm1OutSize = fATilingData->mm1OutSize;
        uint64_t smOnlineOutSize = fATilingData->smOnlineOutSize;
        uint64_t mm2OutSize = fATilingData->mm2OutSize;
        uint32_t batch = fATilingData->batch;
        uint32_t qHeads = fATilingData->numHeads;
        uint32_t kvHeads = fATilingData->kvHeads;
        uint32_t embed = fATilingData->embeddingSize;
        uint32_t embedV = fATilingData->embeddingSizeV;
        uint32_t pagedBlockSize = fATilingData->blockSize;
        uint32_t maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
        uint32_t firstBatchTaskNum = fATilingData->firstBatchTaskNum;
        uint32_t totalTaskNum = fATilingData->totalTaskNum;
        uint32_t blockSize = fATilingData->blockSize;
        uint32_t maskType = fATilingData->maskType;
        float scaleValue = fATilingData->scaleValue;

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK *)params.v);
        AscendC::GlobalTensor<ElementMask> gMask;
        gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
        AscendC::GlobalTensor<int64_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
        AscendC::GlobalTensor<int64_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)(params.s));
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)(params.p));
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.oTemp));
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)(params.oUpdate));
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse *)params.s);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
    #ifdef __DAV_C220_CUBE__
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);

        uint32_t kDynNum = RoundUp(embed, NUM_128);
        kDynNum = kDynNum < NUM_256 ? NUM_256 : kDynNum;
        uint32_t maxQKPL1Size = L1_MAX_SIZE - embedV * MAX_KV_STACK_LEN * sizeof(ElementV);
        uint32_t maxQL1Size = Q_TILE_CEIL * kDynNum * sizeof(ElementQ);
        uint32_t maxNDynNum =
            ((maxQKPL1Size - maxQL1Size) / kDynNum / sizeof(ElementV) / DOUBLE_BUFFER) / NUM_32 * NUM_32;

        uint32_t nDynNum = maxNDynNum < L1_MAX_N_NUM ? maxNDynNum : L1_MAX_N_NUM;
        nDynNum = L1_MAX_N_NUM % nDynNum != 0 ? RoundDown((nDynNum - 1), NUM_32) : nDynNum;

        uint32_t L1_QK_SIZE = BlockMmadQK::L1TileShape::M * kDynNum * sizeof(ElementQ);
        BlockMmadQK blockMmadQK(resource, nDynNum, kDynNum);
        uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;
        BlockMmadPV blockMmadPV(resource, nDynNum, kPVDynNum, L1_QK_SIZE);
    #endif
    #ifdef __DAV_C220_VEC__
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
        EpilogueRescaleO epilogueRescaleO(resource);

        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    #endif
        uint64_t strideQ = qHeads * embed;
        uint64_t strideO = qHeads * embedV;
        uint64_t strideK = kvHeads * embed;
        uint64_t strideV = kvHeads * embedV;
        uint32_t embedRound = RoundUp(embed, FaiKenel::BLOCK_SIZE);
        uint32_t embedRoundV = RoundUp(embedV, FaiKenel::BLOCK_SIZE);
        uint32_t groupSize = qHeads / kvHeads;

        uint64_t qBOffset = 0;
        uint64_t kBOffset = 0;
        uint64_t vBOffset = 0;
        uint64_t oBOffset = 0;
        uint64_t lseBOffset = 0;
        uint64_t blockBOffset = 0;

        uint32_t preTotalTaskNum = 0;
        uint32_t curBatch = 0;
        uint32_t totalQTokens = static_cast<uint32_t>(gActualQseqlen.GetValue(batch - 1));
        uint32_t qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
        uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
        if constexpr(INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
            if (curBatch > 0) {
                uint32_t prevQSeqlenSum = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch - 1));
                qSeqlen = qSeqlen - prevQSeqlenSum;
            }
            if constexpr (!PAGED_CACHE_FLAG) {
                if (curBatch > 0) {
                    uint32_t prevKvSeqlenSum = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch - 1));
                    kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                }
            }
        }
        uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
        uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
        uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;
        uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
        uint32_t curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
        uint32_t curTotalTaskNum = firstBatchTaskNum;

        // Go through each task.
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
            // Get the offset of each core on the GM.
            while (taskIdx >= curTotalTaskNum) {
                ++curBatch;
                preTotalTaskNum = curTotalTaskNum;
                qBOffset += qSeqlen * strideQ;
                if constexpr (!PAGED_CACHE_FLAG) {
                    kBOffset += kvSeqlen * strideK;
                    vBOffset += kvSeqlen * strideV;
                } else {
                    blockBOffset += maxNumBlocksPerBatch;
                }
                oBOffset += qSeqlen * strideO;
                lseBOffset += qSeqlen * qHeads;

                qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
                kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                if constexpr(INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                    if (curBatch > 0) {
                        uint32_t prevQSeqlenSum = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch - 1));
                        qSeqlen = qSeqlen - prevQSeqlenSum;
                    }
                    if constexpr (!PAGED_CACHE_FLAG) {
                        if (curBatch > 0) {
                            uint32_t prevKvSeqlenSum = static_cast<uint32_t>(
                                    gActualKvseqlen.GetValue(curBatch - 1));
                            kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                        }
                    }
                }
                curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                curQSBlockTile = GetQSBlockTile(kvSeqlen);
                curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
                curTotalTaskNum += curQNBlockNum * curQSBlockNum;
            }
            uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
            uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
            uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;
            uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;

            uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
            uint32_t qNStartIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
            uint32_t lseTokenOffset = qSBlockIdx * curQSBlockTile * qHeads;

            uint64_t gmOffsetQ = qBOffset + qSBlockIdx * curQSBlockTile * strideQ + qNStartIdx * embed;
            uint64_t gmOffsetK = kBOffset + kvNIdx * embed;
            uint64_t gmOffsetV = vBOffset + kvNIdx * embedV;
            if constexpr (PAGED_CACHE_FLAG && std::is_same_v<LayoutK, layout::nZ>) {
                gmOffsetK = kBOffset + static_cast<uint64_t>(kvNIdx * embed * pagedBlockSize);
            }
            if constexpr (PAGED_CACHE_FLAG && std::is_same_v<LayoutV, layout::zN>) {
                gmOffsetV = vBOffset + static_cast<uint64_t>(kvNIdx * embedV * pagedBlockSize);
            }
            uint64_t gmOffsetO = oBOffset + qSBlockIdx * curQSBlockTile * strideO + qNStartIdx * embedV;
            uint32_t gmOffsetLse = lseBOffset + lseTokenOffset + qNStartIdx;

            uint32_t qSBlockSize =
                (qSBlockIdx == (curQSBlockNum - 1)) ? (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
            uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1))
                                    ? (groupSize - qNBlockIdxCurGroup * curQNBlockTile)
                                    : curQNBlockTile;
            uint32_t rowNum = qSBlockSize * qNBlockSize;
            uint32_t rowNumRound = RoundUp(rowNum, FaiKenel::BLOCK_SIZE);

            uint32_t noSkipKvS = kvSeqlen;
            if (maskType != 0) {
                uint32_t diffS = kvSeqlen - qSeqlen;
                noSkipKvS = (qSBlockIdx + 1) * curQSBlockTile + diffS;
                noSkipKvS = AscendC::Std::min((uint32_t)kvSeqlen, noSkipKvS);
            }
            uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, pagedBlockSize);

            uint32_t blockStackNum = MAX_KV_STACK_LEN / pagedBlockSize;
            uint32_t stackSeqTile;
            uint32_t stackSeqTilePad = blockStackNum * pagedBlockSize;
            uint32_t preKVNum = PRE_LAUNCH * blockStackNum;
            int32_t stackSeqCount = 0;

    #ifdef __DAV_C220_CUBE__
            LayoutQ layoutQTemp(rowNum, embed);
            uint32_t kRow = strideK;
            uint32_t kCol = blockStackNum * pagedBlockSize;
            uint32_t vRow = blockStackNum * pagedBlockSize;
            uint32_t vCol = strideV;
            if constexpr (PAGED_CACHE_FLAG && std::is_same_v<LayoutK, layout::nZ>) {
                kRow = blockStackNum * strideK;
                kCol = pagedBlockSize;
            }
            if constexpr (PAGED_CACHE_FLAG && std::is_same_v<LayoutV, layout::zN>) {
                vRow = pagedBlockSize;
                vCol = blockStackNum * strideV;
            }
            LayoutK layoutKTemp = LayoutK::template MakeLayout<ElementK>(kRow, kCol);
            LayoutV layoutVTemp = LayoutV::template MakeLayout<ElementV>(vRow, vCol);
            blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qHeads);
    #endif
            for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal + preKVNum; kvSIdx += blockStackNum) {
                if (kvSIdx < kvSLoopNumTotal) {
                    if (kvSIdx + blockStackNum > kvSLoopNumTotal - 1) {
                        stackSeqTile = noSkipKvS - kvSIdx * pagedBlockSize;
                    } else {
                        stackSeqTile = pagedBlockSize * blockStackNum;
                    }
                    uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1);
                    uint64_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                        curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                    LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
    #ifdef __DAV_C220_CUBE__
                    if constexpr (PAGED_CACHE_FLAG) {
                        blockMmadQK(
                            gQ[gmOffsetQ],
                            gK[gmOffsetK],
                            gS[gmOffsetS],
                            gBlockTable[blockBOffset],
                            layoutQTemp,
                            layoutKTemp,
                            layOutS,
                            actualBlockShapeQK,
                            kvSIdx,
                            kvSLoopNumTotal,
                            pagedBlockSize,
                            strideK);
                    } else {
                        blockMmadQK(
                            gQ[gmOffsetQ],
                            gK[gmOffsetK],
                            gS[gmOffsetS],
                            gBlockTable,
                            layoutQTemp,
                            layoutKTemp,
                            layOutS,
                            actualBlockShapeQK,
                            kvSIdx,
                            kvSLoopNumTotal,
                            pagedBlockSize,
                            strideK);
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
    #endif
    #ifdef __DAV_C220_VEC__
                    LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                    LayoutMask layOutMask(2048, 2048, 2048);
                    uint64_t gmOffsetP = gmOffsetS;
                    // causal mask的左上起点
                    uint32_t triUp = noSkipKvS - qSBlockSize;
                    // causal mask的右下止点
                    uint32_t triDown = noSkipKvS;
                    uint32_t kvSStartIdx = kvSIdx * pagedBlockSize;
                    uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;
                    // 在causal mask场景下，由mask的左上起点判断当前基块是否需要加mask
                    // 如果实际加mask长度只有1，那么相当于不加mask（主对角线需要被计算）
                    bool doTriUMask = triUp < kvSEndIdx - 1;
                    if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL) {
                        if (doTriUMask) {
                            epilogueOnlineSoftmax(
                                gP[gmOffsetP],
                                gS[gmOffsetS],
                                gMask,
                                layOutP,
                                layOutS,
                                layOutMask,
                                actualBlockShapeQK,
                                (stackSeqCount == 0),
                                qSBlockSize,
                                qNBlockSize,
                                curStackTileMod,
                                qkReady,
                                triUp,
                                triDown,
                                kvSStartIdx,
                                kvSEndIdx);
                        } else {
                            uint32_t noMaskStackSeqNum = (triUp + 1) / MAX_KV_STACK_LEN;
                            Arch::CrossCoreWaitFlag(qkReady);
                            epilogueOnlineSoftmax(
                                gP[gmOffsetP],
                                gS[gmOffsetS],
                                layOutP,
                                layOutS,
                                actualBlockShapeQK,
                                (stackSeqCount == 0),
                                (stackSeqCount == noMaskStackSeqNum - 1),
                                qSBlockSize,
                                qNBlockSize,
                                curStackTileMod);
                        }
                    } else {
                        Arch::CrossCoreWaitFlag(qkReady);
                        epilogueOnlineSoftmax(
                            gP[gmOffsetP],
                            gS[gmOffsetS],
                            layOutP,
                            layOutS,
                            actualBlockShapeQK,
                            (stackSeqCount == 0),
                            0,
                            qSBlockSize,
                            qNBlockSize,
                            curStackTileMod);
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
    #endif
                }
                if (kvSIdx >= preKVNum) {
                    uint32_t nowkvSIdx = kvSIdx - preKVNum;
                    if (nowkvSIdx + blockStackNum > kvSLoopNumTotal - 1) {
                        stackSeqTile = noSkipKvS - nowkvSIdx * pagedBlockSize;
                    } else {
                        stackSeqTile = pagedBlockSize * blockStackNum;
                    }
                    uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1);
                    uint64_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                    GemmCoord actualBlockShapePV{rowNum, embedV, stackSeqTile};
                    LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);
    #ifdef __DAV_C220_CUBE__
                    LayoutP layoutPTemp(rowNum, stackSeqTile, stackSeqTilePad);
                    uint64_t gmOffsetP = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                        curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;;
                    if constexpr (PAGED_CACHE_FLAG) {
                        blockMmadPV(
                            gP[gmOffsetP],
                            gV[gmOffsetV],
                            gOTmp[gmOffsetOTmp],
                            gBlockTable[blockBOffset],
                            layoutPTemp,
                            layoutVTemp,
                            layoutOTmp,
                            actualBlockShapePV,
                            nowkvSIdx,
                            kvSLoopNumTotal,
                            pagedBlockSize,
                            noSkipKvS,
                            strideV,
                            blockStackNum,
                            softmaxReady);
                    } else {
                        blockMmadPV(
                            gP[gmOffsetP],
                            gV[gmOffsetV],
                            gOTmp[gmOffsetOTmp],
                            gBlockTable,
                            layoutPTemp,
                            layoutVTemp,
                            layoutOTmp,
                            actualBlockShapePV,
                            nowkvSIdx,
                            kvSLoopNumTotal,
                            pagedBlockSize,
                            noSkipKvS,
                            strideV,
                            blockStackNum,
                            softmaxReady);
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
    #endif
    #ifdef __DAV_C220_VEC__
                    LayoutO layoutO(qSeqlen, embed * qHeads);
                    LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                    LayoutLse layoutLse(totalQTokens, qHeads);
                    uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * WORKSPACE_BLOCK_SIZE_DB);

                    Arch::CrossCoreWaitFlag(pvReady);
                    // rescale O
                    epilogueRescaleO(
                        gO[gmOffsetO],
                        gOTmp[gmOffsetOTmp],
                        gOUpdate[gmOffsetUpdate],
                        gLse[gmOffsetLse],
                        layoutO,
                        layoutOTmp,
                        layoutUpdate,
                        layoutLse,
                        actualBlockShapePV,
                        qSBlockSize,
                        qNBlockSize,
                        (stackSeqCount - PRE_LAUNCH == 0),
                        nowkvSIdx + blockStackNum >= kvSLoopNumTotal,
                        curStackTileMod);
    #endif
                }
                stackSeqCount++;
            }
        }
    #ifdef __DAV_C220_CUBE__
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    #endif
    #ifdef __DAV_C220_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
    #endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_ID};
};

template <
    typename InputDtypeQ = half,
    typename InputDtypeKv = half,
    class LAYOUT_K = layout::ColumnMajor,
    class LAYOUT_V = layout::RowMajor,
    bool PagedCacheFlag = false,
    FaiKenel::MaskType maskCategory = FaiKenel::MaskType::NO_MASK,
    FaiKenel::inputLayout inLayout = FaiKenel::inputLayout::TND,
    Epilogue::LseMode lseMode = Epilogue::LseMode::NONE>
__global__ __aicore__ void FAInfer(GM_ADDR q,
                            GM_ADDR k,
                            GM_ADDR v,
                            GM_ADDR mask,
                            GM_ADDR blockTables,
                            GM_ADDR o,
                            GM_ADDR actualQseqlen,
                            GM_ADDR actualKvseqlen,
                            GM_ADDR s,
                            GM_ADDR p,
                            GM_ADDR oTemp,
                            GM_ADDR oUpdate,
                            GM_ADDR tiling
)
{
    using ArchTag = Arch::AtlasA2;
    using ElementQ = InputDtypeQ;
    using LayoutQ = layout::RowMajor;
    using ElementK = InputDtypeKv;
    using LayoutK = LAYOUT_K;
    using ElementV = InputDtypeKv;
    using LayoutV = LAYOUT_V;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = InputDtypeQ;
    using LayoutP = layout::RowMajor;
    using ElementO = InputDtypeQ;
    using LayoutO = layout::RowMajor;
    using ElementLse = float;
    using LayoutLse = layout::RowMajor;
    using ElementMask = int8_t;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;

    using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    using DispatchPolicyQK = Gemm::MmadAtlasA2XFAIQK<PagedCacheFlag, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2XFAIOnlineSoftmax<lseMode>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax =
        Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;

    using L1TileShapePV = GemmShape<128, 128, 256>;
    using L0TileShapePV = GemmShape<128, 128, 128>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2XFAIPV<PagedCacheFlag, false>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2XFAIRescaleO<lseMode>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;

    using FAInferKernel = FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO,
                                        PagedCacheFlag, maskCategory, inLayout>;
    FAIKernelParams params{q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen, o, s, p, oTemp, oUpdate, tiling};
    FAInferKernel flashAttnInfer;
    flashAttnInfer(params);
}

#endif // _ASCENDC_X_FLASH_ATTENTION_INFER_H_
