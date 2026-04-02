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

#ifndef X_ATTN_CATLASS_KERNEL_H
#define X_ATTN_CATLASS_KERNEL_H

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_common.hpp"
#include "kernel_operator.h"
using namespace Catlass;
 
template <
    class BlockMmadQK,
    class BlockMmadPV,
    class EpilogueFAUnsharedSoftmax,
    bool PAGED_CACHE_FLAG
    >
class UnsharedFAInferKernel {
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
    using ElementO = typename BlockMmadPV::ElementC;
    using LayoutO = typename BlockMmadPV::LayoutC;

    CATLASS_DEVICE
    UnsharedFAInferKernel(XAttentionTilingData* tilingDataPtr): faTilingData(tilingDataPtr) {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(XAttnKernelParams const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(XAttnKernelParams const &params) {
        static constexpr uint32_t L1_QK_SIZE = BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K
                                                   * sizeof(ElementQ)
                                               + BlockMmadQK::L1TileShape::N * BlockMmadQK::L1TileShape::K
                                                     * sizeof(ElementK) * 2;

        BlockMmadQK blockMmadQK(resource);
        BlockMmadPV blockMmadPV(resource, L1_QK_SIZE);

        // __gm__ XATilingData *faTilingData = reinterpret_cast<__gm__ XATilingData *>(params.tiling);
        
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gUnsharedK;
        gUnsharedK.SetGlobalBuffer((__gm__ ElementK *)params.unshared_k);
        AscendC::GlobalTensor<ElementV> gUnsharedV;
        gUnsharedV.SetGlobalBuffer((__gm__ ElementV *)params.unshared_v);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
        AscendC::GlobalTensor<ElementO> gUnsharedO;
        gUnsharedO.SetGlobalBuffer((__gm__ ElementO *)params.unshared_workspace);
        AscendC::GlobalTensor<int32_t> gUnsharedBlockTable;
        gUnsharedBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.unsharedBlockTable));

        uint32_t maxDecodeStep = faTilingData->maxDecodeStep;
        uint32_t embeddingSize = faTilingData->embeddingSize;
        uint32_t groupSize = faTilingData->groupSize;
        uint32_t unsharedCoreNum = faTilingData->unsharedCoreNum;
        uint32_t unshareGroupCountPerLoop = faTilingData->unshareGroupCountPerLoop;
        uint32_t unsharedTaskNumHead = faTilingData->unsharedTaskNumHead;
        uint32_t unsharedTaskNumTail = faTilingData->unsharedTaskNumTail;
        uint32_t unsharedFullCoreNum = faTilingData->unsharedFullCoreNum;
        uint32_t unsharedLoopCountPerBatch = faTilingData->unsharedLoopCountPerBatch;
        uint32_t coreNumShared = faTilingData->sharedCoreNum;

        uint32_t relativeCoreIdx = AscendC::GetBlockIdx() - coreNumShared;
        uint32_t taskStartIdx = relativeCoreIdx <= unsharedFullCoreNum ? relativeCoreIdx * unsharedTaskNumHead:
                                (unsharedFullCoreNum * unsharedTaskNumHead + (relativeCoreIdx - unsharedFullCoreNum) * unsharedTaskNumTail) ;
        uint32_t taskEndIdx = taskStartIdx + (relativeCoreIdx >= unsharedFullCoreNum ? unsharedTaskNumTail : unsharedTaskNumHead);
        //cce::printf("aic blockIdx:%d, sub block num:%d, loopCount:%d\n", coreIdx, AscendC::GetSubBlockNum(), taskEndIdx - taskStartIdx);
        
        uint64_t qOBaseBlockSize = groupSize * unshareGroupCountPerLoop * embeddingSize;
        uint64_t kvBaseBlockSize = maxDecodeStep * unshareGroupCountPerLoop * embeddingSize;
        uint64_t gmQOffset = taskStartIdx * qOBaseBlockSize;
        uint64_t gmOOffset = gmQOffset;
        uint64_t gmKOffset = 0;
        GetKVOffset(taskStartIdx, unsharedLoopCountPerBatch, kvBaseBlockSize, gUnsharedBlockTable, gmKOffset);
        uint64_t gmVOffset = gmKOffset;
        uint64_t gmSOffset = (coreNumShared * WORKSPACE_BLOCK_SIZE_DB +
                              relativeCoreIdx * UNSHARED_WORKSPACE_BLOCK_SIZE_DB) * NUM3;
        uint64_t gmPOffset = gmSOffset;
        uint64_t sRelativeOffset = 0;
        uint64_t pRelativeOffset = 0;
        uint64_t actualPOffset = 0;
        uint64_t actualSOffset = 0;

        LayoutQ layoutQTemp(unshareGroupCountPerLoop * groupSize, embeddingSize);
        LayoutK layoutKTemp(embeddingSize, unshareGroupCountPerLoop * maxDecodeStep);
        LayoutV layoutVTemp(unshareGroupCountPerLoop * maxDecodeStep, embeddingSize);
        LayoutO layoutOTemp(unshareGroupCountPerLoop * groupSize, embeddingSize);
        LayoutS layoutSTemp(unshareGroupCountPerLoop * groupSize, unshareGroupCountPerLoop * maxDecodeStep);
        LayoutP layoutPTemp(unshareGroupCountPerLoop * groupSize, unshareGroupCountPerLoop * maxDecodeStep);
        GemmCoord actualBlockShapeQK{unshareGroupCountPerLoop * groupSize, unshareGroupCountPerLoop * maxDecodeStep,
                                     embeddingSize};
        GemmCoord actualBlockShapePV{unshareGroupCountPerLoop * groupSize, embeddingSize,
                                     unshareGroupCountPerLoop * maxDecodeStep};
        for (uint32_t taskIdx = taskStartIdx, nextTaskIdx; taskIdx < taskEndIdx + PRE_LAUNCH; ++taskIdx) {
            if (taskIdx < taskEndIdx) {
                sRelativeOffset = (taskIdx % (PRE_LAUNCH + 1)) * UNSHARED_WORKSPACE_BLOCK_SIZE_DB;
                actualSOffset = gmSOffset + sRelativeOffset;
                if (taskIdx >= taskStartIdx + PRE_LAUNCH) {
                    AscendC::PipeBarrier<PIPE_MTE2>();
                }
                blockMmadQK(
                    gQ[gmQOffset], gUnsharedK[gmKOffset], gS[actualSOffset],
                    layoutQTemp, layoutKTemp, layoutSTemp, actualBlockShapeQK);
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                gmQOffset += qOBaseBlockSize;
                nextTaskIdx = taskIdx + 1;
                if (nextTaskIdx % unsharedLoopCountPerBatch == 0) {
                    GetKVOffset(nextTaskIdx, unsharedLoopCountPerBatch, kvBaseBlockSize, gUnsharedBlockTable,
                                gmKOffset);
                } else {
                    gmKOffset += kvBaseBlockSize;
                } 
            }
            if (taskIdx >= taskStartIdx + PRE_LAUNCH) {
                pRelativeOffset = ((taskIdx - PRE_LAUNCH) % (PRE_LAUNCH + 1)) * UNSHARED_WORKSPACE_BLOCK_SIZE_DB;
                actualPOffset = gmPOffset + pRelativeOffset;
                blockMmadPV(
                    gP[actualPOffset],
                    gUnsharedV[gmVOffset], gUnsharedO[gmOOffset],
                    layoutPTemp, layoutVTemp, layoutOTemp,
                    actualBlockShapePV, softmaxReady);
                gmOOffset += qOBaseBlockSize;
                nextTaskIdx = taskIdx + 1 - PRE_LAUNCH;
                if (nextTaskIdx % unsharedLoopCountPerBatch == 0) {
                    GetKVOffset(nextTaskIdx, unsharedLoopCountPerBatch, kvBaseBlockSize, gUnsharedBlockTable,
                                gmVOffset);
                } else {
                    gmVOffset += kvBaseBlockSize;
                }
            }
        }
    }
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(XAttnKernelParams const &params) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
        AscendC::GlobalTensor<uint32_t> gDecodeStep;
        gDecodeStep.SetGlobalBuffer((__gm__ uint32_t *)params.decodeStep);

        uint32_t batch = faTilingData->batch;
        uint32_t beamSize = faTilingData->beamSize;
        uint32_t numHeads = faTilingData->numHeads;
        uint32_t unsharedKvSeqLen = gDecodeStep.GetValue(0);
        uint32_t maxDecodeStep = faTilingData->maxDecodeStep;
        uint32_t embeddingSize = faTilingData->embeddingSize;
        uint32_t groupSize = faTilingData->groupSize;
        uint32_t unsharedCoreNum = faTilingData->unsharedCoreNum;
        uint32_t unshareGroupCountPerLoop = faTilingData->unshareGroupCountPerLoop;
        uint32_t unsharedTaskNumHead = faTilingData->unsharedTaskNumHead;
        uint32_t unsharedTaskNumTail = faTilingData->unsharedTaskNumTail;
        uint32_t unsharedFullCoreNum = faTilingData->unsharedFullCoreNum;
        float scaleValue = faTilingData->scaleValue;
        uint32_t coreNumShared = faTilingData->sharedCoreNum;

        uint32_t gUnsharedOffset = batch * beamSize * numHeads * embeddingSize;
        AscendC::GlobalTensor<ElementO> gUnsharedGm;
        gUnsharedGm.SetGlobalBuffer(((__gm__ ElementO *)params.unshared_workspace) + gUnsharedOffset);
        gUnsharedOffset += batch * beamSize * numHeads;
        AscendC::GlobalTensor<ElementO> gUnsharedGl;
        gUnsharedGl.SetGlobalBuffer(((__gm__ ElementO *)params.unshared_workspace) + gUnsharedOffset);

        uint32_t relativeCoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum() - coreNumShared;
        uint32_t taskStartIdx = relativeCoreIdx <= unsharedFullCoreNum ? relativeCoreIdx * unsharedTaskNumHead:
                                (unsharedFullCoreNum * unsharedTaskNumHead + (relativeCoreIdx - unsharedFullCoreNum) * unsharedTaskNumTail) ;
        uint32_t taskEndIdx = taskStartIdx + (relativeCoreIdx >= unsharedFullCoreNum ? unsharedTaskNumTail : unsharedTaskNumHead);
        
        EpilogueFAUnsharedSoftmax epilogueFAUnsharedSoftmax(resource, scaleValue, unsharedKvSeqLen, maxDecodeStep, unshareGroupCountPerLoop,
                                                            groupSize);
   
        uint64_t gmSPOffset = (coreNumShared * WORKSPACE_BLOCK_SIZE_DB  +
                               relativeCoreIdx * UNSHARED_WORKSPACE_BLOCK_SIZE_DB) * NUM3;
        uint64_t spRelativeOffset = 0;
        uint64_t actualSPOffset = 0;
        int64_t gmGlBaseBlockSize = groupSize * unshareGroupCountPerLoop;
        uint64_t gmUnsharedGmGlOffset = taskStartIdx * gmGlBaseBlockSize;
        LayoutS layoutSTemp(unshareGroupCountPerLoop * groupSize, unshareGroupCountPerLoop * maxDecodeStep);
        LayoutP layoutPTemp(unshareGroupCountPerLoop * groupSize, unshareGroupCountPerLoop * maxDecodeStep);
        GemmCoord actualBlockShapeQK{unshareGroupCountPerLoop * groupSize, unshareGroupCountPerLoop * maxDecodeStep,
                                     embeddingSize};
        for (uint32_t taskIdx = taskStartIdx; taskIdx < taskEndIdx; ++taskIdx) {
            spRelativeOffset = (taskIdx % (PRE_LAUNCH + 1)) * UNSHARED_WORKSPACE_BLOCK_SIZE_DB;
            actualSPOffset = gmSPOffset + spRelativeOffset;
            Arch::CrossCoreWaitFlag(qkReady);
            // FA unshared softmax
            epilogueFAUnsharedSoftmax(
                gP[actualSPOffset], gS[actualSPOffset], gUnsharedGm[gmUnsharedGmGlOffset],
                gUnsharedGl[gmUnsharedGmGlOffset], layoutPTemp, layoutSTemp,
                actualBlockShapeQK, unshareGroupCountPerLoop
            );
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
            gmUnsharedGmGlOffset += gmGlBaseBlockSize;
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
  private:
    CATLASS_DEVICE void GetKVOffset(uint32_t taskIdx, uint32_t unsharedLoopCountPerBatch, uint64_t kvBaseBlockSize,
                                    const AscendC::GlobalTensor<int32_t> &gUnsharedBlockTable, uint64_t &offset) {
        if constexpr(PAGED_CACHE_FLAG) {
            uint32_t batchId = gUnsharedBlockTable.GetValue(taskIdx / unsharedLoopCountPerBatch);
            uint32_t remainCnt = taskIdx % unsharedLoopCountPerBatch;
            offset = (batchId * unsharedLoopCountPerBatch + remainCnt) * kvBaseBlockSize;
        } else {
            offset = taskIdx * kvBaseBlockSize;
        }
    }
  private:
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
    XAttentionTilingData* faTilingData;
};
 
 /*
     FASharedInferKernel
     Compute Stream
     1. BlockMmadQK
     2. OnlineSoftmax
     3. BlockMmadPV
     4. EpilogueRescaleO 最后rescaleO 不需要div rowsum
 */
 template <
     class BlockMmadQK,
     class BlockMmadPV,
     class BlockMmadQKTail,
     class BlockMmadPVTail,
     class EpilogueOnlineSoftmax,
     class EpilogueRescaleO,
     bool PAGED_CACHE_FLAG = true
 >
 class SharedFAInferKernel {
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
 
     // Methods
     CATLASS_DEVICE
     SharedFAInferKernel(XAttentionTilingData* tilingDataPtr): faTilingData(tilingDataPtr) {
     }
 
     template <int32_t CORE_TYPE = g_coreType>
     CATLASS_DEVICE void operator()(XAttnKernelParams const &params);
 
     template <>
     CATLASS_DEVICE void operator()<AscendC::AIC>(XAttnKernelParams const &params) {
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
         static constexpr uint32_t L1_QK_SIZE = BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K
                                                    * sizeof(ElementQ)
                                                + BlockMmadQK::L1TileShape::N * BlockMmadQK::L1TileShape::K
                                                      * sizeof(ElementK) * 2;
         BlockMmadQK blockMmadQK(resource);
         BlockMmadPV blockMmadPV(resource, L1_QK_SIZE);
 
         BlockMmadQKTail blockMmadQKTail(resource);
         BlockMmadPVTail blockMmadPVTail(resource, L1_QK_SIZE);
         // __gm__ XATilingData *faTilingData = reinterpret_cast<__gm__ XATilingData *>(params.tiling);
         uint64_t mm1OutSize = faTilingData->mm1OutSize;
         uint64_t smOnlineOutSize = faTilingData->smOnlineOutSize;
         uint32_t batch = faTilingData->batch;  // requestNum 
         uint32_t beamSize = faTilingData->beamSize; 
         uint32_t qHeads = faTilingData->numHeads;
         uint32_t kvHeads = faTilingData->kvHeads;
         uint32_t embed = faTilingData->embeddingSize;
         uint32_t pagedBlockSize = faTilingData->blockSize;
         uint32_t sharedCoreNum = faTilingData->sharedCoreNum;
         uint32_t maxNumBlocksPerBatch = faTilingData->maxNumBlocksPerBatch;
         uint32_t curTotalTaskNum = faTilingData->firstSharedBatchTaskNum;
         uint32_t totalTaskNum = faTilingData->sharedTotalTaskNum;
         uint32_t blockSize = faTilingData->blockSize;
         uint32_t maskType = faTilingData->maskType;
         float scaleValue = faTilingData->scaleValue;
 
         AscendC::GlobalTensor<ElementQ> gQ;
         gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
         AscendC::GlobalTensor<ElementK> gK;
         gK.SetGlobalBuffer((__gm__ ElementK *)params.k_cache);
         AscendC::GlobalTensor<ElementK> gV;
         gV.SetGlobalBuffer((__gm__ ElementK *)params.v_cache);
         AscendC::GlobalTensor<int32_t> gBlockTable;
         gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.sharedBlockTable));
         AscendC::GlobalTensor<int32_t> gActualKvseqlen;
         gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualKvseqlen);
         AscendC::GlobalTensor<ElementS> gS;
         gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
         AscendC::GlobalTensor<ElementP> gP;
         gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
         AscendC::GlobalTensor<ElementOTmp> gOTmp;
         gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTemp);
 
         uint64_t strideQO = qHeads * embed;
         uint64_t strideKV = kvHeads * embed;
         uint32_t embedRound = RoundUp<BLOCK_SIZE>(embed);
         uint32_t groupSize = qHeads / kvHeads;
 
         uint32_t coreIdx = AscendC::GetBlockIdx();
         uint32_t coreNum = sharedCoreNum; // TODO coreNum Need To be modified
         curTotalTaskNum = 0;
         uint32_t preTotalTaskNum = 0;
         uint32_t curBatch = 0;
         uint64_t qBOffset = 0;
         uint64_t kBOffset = 0;
         uint64_t vBOffset = 0;
         uint64_t blockBOffset = 0;
         int64_t qSeqlen = 0;
         int64_t kvSeqlen = 0;
         uint32_t curQNBlockTile;
         uint32_t qNBlockNumPerGroup;
         uint32_t curQNBlockNum;
         int64_t curQSBlockTile;
         uint32_t curQSBlockNum;
 
         preTotalTaskNum = curTotalTaskNum;
         qSeqlen = beamSize;
         kvSeqlen = gActualKvseqlen.GetValue(curBatch);
         curQSBlockTile = GetQSBlockTile(kvSeqlen);
         curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
         qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
         curQNBlockNum = qNBlockNumPerGroup * kvHeads;
         curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
         curTotalTaskNum += curQNBlockNum * curQSBlockNum;
         for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
             while (taskIdx >= curTotalTaskNum) {
                 ++curBatch;
                 preTotalTaskNum = curTotalTaskNum;
                 qBOffset += qSeqlen * strideQO;
                 if constexpr (!PAGED_CACHE_FLAG) {
                     kBOffset += kvSeqlen * strideKV;
                     vBOffset += kvSeqlen * strideKV;
                 } else {
                     blockBOffset += maxNumBlocksPerBatch;
                 }
                 qSeqlen = beamSize;
                 kvSeqlen = gActualKvseqlen.GetValue(curBatch);
                 curQSBlockTile = GetQSBlockTile(kvSeqlen);
                 curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                 qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                 curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                 curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
                 curTotalTaskNum += curQNBlockNum * curQSBlockNum;
             }
             uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
             uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
             uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;
             uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
             uint32_t kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
             uint32_t qHeadIdx = kvHeadIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
             uint64_t gmQOffset = qBOffset + qSBlockIdx * curQSBlockTile * strideQO + qHeadIdx * embed;
             uint64_t gmKOffset = kBOffset + kvHeadIdx * embed;
             uint64_t gmVOffset = vBOffset + kvHeadIdx * embed;
             uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1)) ? (qSeqlen - qSBlockIdx * curQSBlockTile)
                                                                        : curQSBlockTile;
             uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1))
                                        ? (groupSize - qNBlockIdxCurGroup * curQNBlockTile)
                                        : curQNBlockTile;
             uint32_t rowNum = qSBlockSize * qNBlockSize;
             uint32_t rowNumRound = AlignUp(rowNum, BLOCK_SIZE);
             uint32_t noSkipKvS = kvSeqlen;
             uint32_t noMaskKvS = kvSeqlen;
             uint32_t noMaskTailS = 0;
             // if (maskType != 0) {
             //     uint32_t diffS = kvSeqlen - qSeqlen;
             //     noSkipKvS = (qSBlockIdx + 1) * curQSBlockTile + diffS;
             //     noSkipKvS = Min((uint32_t)kvSeqlen, noSkipKvS);
             //     noMaskKvS = noSkipKvS - qSBlockSize;
             //     noMaskTailS = noMaskKvS % pagedBlockSize;
             // }
             uint32_t maskedKvS = qSBlockSize;
             uint32_t kvSLoopNumNoMask = CeilDiv(noMaskKvS, pagedBlockSize);
             uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, pagedBlockSize);
             uint32_t blockStackNum = 4;
             uint32_t stackSeqTile;
             uint32_t stackSeqTileRound = blockStackNum * 128;
             int32_t preLaunch = 2;
             int32_t totalStackSeqNum = (maskType != 0) ? (CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize) + 1)
                                                        : CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize);
             int32_t stackSeqCount = 0;
 
             LayoutQ layoutQTemp(rowNum, embed);
             LayoutK layoutKTemp(strideKV, blockStackNum * pagedBlockSize);
             LayoutV layoutVTemp(blockStackNum * pagedBlockSize, strideKV);
             blockMmadQK.loadQGM(gQ[gmQOffset], layoutQTemp, rowNum, qNBlockSize, qHeads);
             for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumNoMask; kvSIdx += blockStackNum) {
                 if (kvSIdx < kvSLoopNumNoMask) {
                     if (kvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                         stackSeqTile = noMaskKvS - kvSIdx * pagedBlockSize;
                     } else {
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     uint32_t SWorkSpacePingPongFlag = stackSeqCount % (preLaunch + 1);
                     uint64_t gmSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1)
                                          + SWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                     GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                     if constexpr (!PAGED_CACHE_FLAG) {
                         blockMmadQK(
                             gQ[gmQOffset], gK[gmKOffset], gS[gmSOffset], gBlockTable, layoutQTemp, layoutKTemp,
                             actualBlockShapeQK, kvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS, strideKV
                         );
                     } else {
                         blockMmadQK(
                             gQ[gmQOffset], gK[gmKOffset], gS[gmSOffset], gBlockTable[blockBOffset], layoutQTemp,
                             layoutKTemp, actualBlockShapeQK, kvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS,
                             strideKV
                         );
                     }
                     Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                 }
                 if (kvSIdx >= preLaunch * blockStackNum) {
                     uint32_t nowkvSIdx = kvSIdx - preLaunch * blockStackNum;
                     if (nowkvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                         stackSeqTile = noMaskKvS - nowkvSIdx * pagedBlockSize;
                     } else {
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     uint32_t PVWorkSpacePingPongFlag = (stackSeqCount - preLaunch) % (preLaunch + 1);
                     uint64_t gmPOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1)
                                          + PVWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                     uint64_t gmOTmpOffset = gmPOffset;
                     LayoutP layoutPTemp(rowNum, stackSeqTileRound);
                     GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                     if constexpr (!PAGED_CACHE_FLAG) {
                         blockMmadPV(
                             gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset], gBlockTable, layoutPTemp, layoutVTemp,
                             actualBlockShapePV, nowkvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS, strideKV,
                             softmaxReady
                         );
                     } else {
                         blockMmadPV(
                             gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset], gBlockTable[blockBOffset], layoutPTemp,
                             layoutVTemp, actualBlockShapePV, nowkvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS,
                             strideKV, softmaxReady
                         );
                     }
                     Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                 }
                 stackSeqCount++;
             }
 
             /*
              * for the secondary loop
              * while masked, it deals the CV stage1(Qk^t/SMOnline) of the final base block(typical shape [128, 512]),
              * and the CV stage2(PV/rescaleO) of the last (prelaunch+1) base blocks while not masked, it deals only the
              * CV stage2(PV/rescaleO) of the last (prelaunch) base blocks
              */
 
             // deal secondary loop conditions
             uint32_t maskedStartIdx = (maskType != 0) ? ((noMaskTailS != 0) ? (kvSLoopNumNoMask - 1) : kvSLoopNumNoMask)
                                                       : AlignUp(kvSLoopNumNoMask, blockStackNum);
             uint32_t noMaskTailInteStackNum = (noMaskKvS / pagedBlockSize) % blockStackNum;
             noMaskTailInteStackNum = (noMaskTailInteStackNum != 0) ? noMaskTailInteStackNum
                                                                    : ((noMaskTailS != 0) ? 0 : blockStackNum);
             uint32_t preLaunchStackNum = (maskType != 0) ? ((preLaunch - 1) * blockStackNum + noMaskTailInteStackNum)
                                                          : (preLaunch * blockStackNum);
 
             // masked kvSeqlen loop 
    
             for (uint32_t kvSIdx = maskedStartIdx; kvSIdx < kvSLoopNumTotal + preLaunchStackNum;) {
                 if ((kvSIdx < kvSLoopNumTotal) && (stackSeqCount <= totalStackSeqNum - 1)) {
                     stackSeqTile = maskedKvS;
                     uint32_t SWorkSpacePingPongFlag = stackSeqCount % (preLaunch + 1);
                     uint64_t gmSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1)
                                          + SWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                     GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                     if constexpr (!PAGED_CACHE_FLAG) {
                         blockMmadQKTail(
                             gQ[gmQOffset], gK[gmKOffset], gS[gmSOffset], gBlockTable, layoutQTemp, layoutKTemp,
                             actualBlockShapeQK, kvSIdx, kvSLoopNumTotal, pagedBlockSize, noSkipKvS, strideKV,
                             noMaskTailS, 1
                         );
                     } else {
                         blockMmadQKTail(
                             gQ[gmQOffset], gK[gmKOffset], gS[gmSOffset], gBlockTable[blockBOffset], layoutQTemp,
                             layoutKTemp, actualBlockShapeQK, kvSIdx, kvSLoopNumTotal, pagedBlockSize, noSkipKvS,
                             strideKV, noMaskTailS, 1
                         );
                     }
                     Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                 }
 
                 if (kvSIdx >= preLaunchStackNum) {
                     uint32_t delayedKvSIdx = kvSIdx - preLaunchStackNum;
 
                     if (delayedKvSIdx + blockStackNum > kvSLoopNumTotal - 1 && (maskType != 0)) {
                         stackSeqTile = maskedKvS;
                     } else if (delayedKvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                         stackSeqTile = noMaskKvS - delayedKvSIdx * pagedBlockSize;
                     } else {
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     uint32_t PVWorkSpacePingPongFlag = (stackSeqCount - preLaunch) % (preLaunch + 1);
                     uint64_t gmPOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1)
                                          + PVWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                     uint64_t gmOTmpOffset = gmPOffset;
                     LayoutP layoutPTemp(rowNum, stackSeqTileRound);
                     GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
 
                     if ((stackSeqCount - preLaunch == totalStackSeqNum - 1) && (maskType != 0)) { // 加mask
                         if constexpr (!PAGED_CACHE_FLAG) {
                             blockMmadPVTail(
                                 gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset], gBlockTable, layoutPTemp,
                                 layoutVTemp, actualBlockShapePV, delayedKvSIdx, kvSLoopNumTotal, pagedBlockSize,
                                 noSkipKvS, strideKV, softmaxReady, noMaskTailS, 1
                             );
                         } else {
                             blockMmadPVTail(
                                 gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset], gBlockTable[blockBOffset],
                                 layoutPTemp, layoutVTemp, actualBlockShapePV, delayedKvSIdx, kvSLoopNumTotal,
                                 pagedBlockSize, noSkipKvS, strideKV, softmaxReady, noMaskTailS, 1
                             );
                         }
                     } else { // 不加mask
                         if constexpr (!PAGED_CACHE_FLAG) {
                             blockMmadPV(
                                 gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset], gBlockTable, layoutPTemp,
                                 layoutVTemp, actualBlockShapePV, delayedKvSIdx, kvSLoopNumNoMask, pagedBlockSize,
                                 noMaskKvS, strideKV, softmaxReady
                             );
                         } else {
                             blockMmadPV(
                                 gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset], gBlockTable[blockBOffset],
                                 layoutPTemp, layoutVTemp, actualBlockShapePV, delayedKvSIdx, kvSLoopNumNoMask,
                                 pagedBlockSize, noMaskKvS, strideKV, softmaxReady
                             );
                         }
                     }
                     Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                 }
                 if ((maskType != 0) && (stackSeqCount - preLaunch == totalStackSeqNum - 2)) {
                     kvSIdx += noMaskTailInteStackNum;
                 } else {
                     kvSIdx += blockStackNum;
                 }
                 stackSeqCount++;
             }
         }
 
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
     }
 
 
     template <>
     CATLASS_DEVICE void operator()<AscendC::AIV>(XAttnKernelParams const &params) {
         AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
 
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
 
         // Get tiling parameters
         // __gm__ XATilingData *faTilingData = reinterpret_cast<__gm__ XATilingData *>(params.tiling);
         uint64_t mm1OutSize = faTilingData->mm1OutSize;
         uint64_t smOnlineOutSize = faTilingData->smOnlineOutSize;
         uint64_t mm2OutSize = faTilingData->mm2OutSize;
         uint32_t batch = faTilingData->batch;
         uint32_t beamSize = faTilingData->beamSize;
         uint32_t qHeads = faTilingData->numHeads;
         uint32_t kvHeads = faTilingData->kvHeads;
         uint32_t embed = faTilingData->embeddingSize;
         uint32_t sharedCoreNum = faTilingData->sharedCoreNum;
         uint32_t pagedBlockSize = faTilingData->blockSize;
         uint32_t maxNumBlocksPerBatch = faTilingData->maxNumBlocksPerBatch;
         uint32_t firstBatchTaskNum = faTilingData->firstSharedBatchTaskNum;
         uint32_t totalTaskNum = faTilingData->sharedTotalTaskNum;
         uint32_t maskType = faTilingData->maskType;
         float scaleValue = faTilingData->scaleValue;
         
         uint64_t gOffsetTempO = batch * beamSize * qHeads * embed;
         uint64_t gMaxOffset = batch * beamSize * qHeads * SOFTMAX_BROAD_SIZE;;
         // Get the memory offset address of the input on Global Memory
         AscendC::GlobalTensor<int32_t> gActualKvseqlen;
         gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualKvseqlen);
         AscendC::GlobalTensor<ElementO> gO;
         gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
         AscendC::GlobalTensor<ElementS> gS;
         gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
         AscendC::GlobalTensor<ElementP> gP;
         gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
         AscendC::GlobalTensor<ElementOTmp> gOTmp;
         gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTemp);
         AscendC::GlobalTensor<ElementOTmp> gOUpdate;
         gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)params.oUpdate);
 
         // shared Gm and Gl output
         AscendC::GlobalTensor<float> gSharedO;
         AscendC::GlobalTensor<ElementS> gSharedSum;
         AscendC::GlobalTensor<ElementS> gSharedMax;
         gSharedO.SetGlobalBuffer((__gm__ float *)params.shared_workspace);
         gSharedMax.SetGlobalBuffer((__gm__ ElementS *)params.shared_workspace + gOffsetTempO);
         gSharedSum.SetGlobalBuffer((__gm__ ElementS *)params.shared_workspace + gOffsetTempO + gMaxOffset);
         

         uint32_t groupSize = qHeads / kvHeads;
         uint32_t embedRound = RoundUp(embed, BLOCK_SIZE);
 
         EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
         EpilogueRescaleO epilogueRescaleO(resource);
 
         // uint32_t curTotalTaskNum = 0;
         uint32_t preTotalTaskNum = 0;
         uint32_t curBatch = 0;
         uint32_t oBatchOffset = 0;
         uint32_t qSeqlen = static_cast<uint32_t>(beamSize);
         uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
         uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
         uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
         uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;
         uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
         uint32_t curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
         uint32_t curTotalTaskNum = firstBatchTaskNum;
 
         uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
         // coreNum Need to be changed
         uint32_t coreNum = sharedCoreNum;

         // Go through each task.
         for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
             // Get the offset of each core on the GM.
             while (taskIdx >= curTotalTaskNum) {
                 curBatch++;
                 oBatchOffset += qSeqlen * qHeads * embed;
                 preTotalTaskNum = curTotalTaskNum;
                 qSeqlen = static_cast<uint32_t>(beamSize);
                 kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                 curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                 qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                 curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                 curQSBlockTile = GetQSBlockTile(kvSeqlen);
                 curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
                 curTotalTaskNum += curQNBlockNum * curQSBlockNum;
             }
             uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
             uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
             uint32_t qNBlockIdx = taskIdxCurBatch % curQNBlockNum;
             uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
 
             uint32_t oSOffset = qSBlockIdx * curQSBlockTile * qHeads * embed;
             uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
             uint32_t qStartNIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
             uint32_t oNOffset = qStartNIdx * embed;
             uint32_t gmOffsetO = oBatchOffset + oSOffset + oNOffset;

             // shared sum max workspace offset
             // Calculate shared workspace offset for this core's task
             // gsharedout size: [batch*beamSize, numHeads, 8]
             // Each core handles: [qSBlockSize, qNBlockSize] elements
             // Offset = curBatch * beamSize * qHeads * 8 + qSBlockIdx * curQSBlockTile * qHeads * 8 + qStartNIdx * 8;
             uint32_t gSharedOffset = curBatch * qSeqlen * qHeads * SOFTMAX_BROAD_SIZE +
                 qSBlockIdx * curQSBlockTile * qHeads * SOFTMAX_BROAD_SIZE + qStartNIdx * SOFTMAX_BROAD_SIZE;
             // cce::printf("gSharedOffset:%d\n", gSharedOffset);
             uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1)) ? (qSeqlen - qSBlockIdx * curQSBlockTile)
                                                                        : curQSBlockTile;
             uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1))
                                        ? (groupSize - qNBlockIdxCurGroup * curQNBlockTile)
                                        : curQNBlockTile;
             uint32_t rowNum = qSBlockSize * qNBlockSize;
             uint32_t rowNumRound = RoundUp(rowNum, BLOCK_SIZE);
 
             uint32_t noSkipKvS = kvSeqlen;
             uint32_t noMaskKvS = kvSeqlen;
             uint32_t noMaskTailS = 0;
             if (maskType != 0) {
                 uint32_t diffS = kvSeqlen - qSeqlen;
                 noSkipKvS = (qSBlockIdx + 1) * curQSBlockTile + diffS;
                 noSkipKvS = Min(kvSeqlen, noSkipKvS);
                 noMaskKvS = noSkipKvS - qSBlockSize;
                 noMaskTailS = noMaskKvS % pagedBlockSize;
             }
             uint32_t maskedKvS = qSBlockSize;
             uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, pagedBlockSize);
             uint32_t kvSLoopNumNoMask = CeilDiv(noMaskKvS, pagedBlockSize);
             uint32_t blockStackNum = 4;
             uint32_t stackSeqTilePad = blockStackNum * pagedBlockSize;
             uint32_t stackSeqTile;
             int32_t preLaunch = 2;
             // totalStackSeqNum = 1
             int32_t totalStackSeqNum = (maskType != 0) ? (CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize) + 1)
                                                        : CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize);
             int32_t stackSeqCount = 0;
 
             // no mask kvSeqlen loop
             for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumNoMask; kvSIdx += blockStackNum) {
 
                 if (kvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                     stackSeqTile = noMaskKvS - kvSIdx * pagedBlockSize;
                 } else {
                     stackSeqTile = pagedBlockSize * blockStackNum;
                 }
                 uint32_t isLastStackTile = (kvSIdx + blockStackNum > kvSLoopNumNoMask - 1) ? 1 : 0;
                 uint32_t stackSeqTileRound = RoundUp(stackSeqTile, BLOCK_SIZE);
                 LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
                 LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                 GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                 uint32_t curStackTileMod = stackSeqCount % (preLaunch + 1);
                 uint32_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + // cube core offset
                                      curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;            // single cube core db offset
                 // vec core offset will be processed within epilogue block
                 uint32_t gmOffsetP = gmOffsetS;
                 // AscendC::printf("stackSeqCount:%d\n", stackSeqCount);
                 Arch::CrossCoreWaitFlag(qkReady);
                 
                 // online softmax
                 epilogueOnlineSoftmax(
                     gP[gmOffsetP], gS[gmOffsetS], gSharedMax[gSharedOffset], gSharedSum[gSharedOffset], layOutP, layOutS, actualBlockShapeQK, (stackSeqCount == 0),
                     isLastStackTile, qSBlockSize, qNBlockSize, curStackTileMod, qHeads
                 );
                 Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
 
                 if (kvSIdx >= preLaunch * blockStackNum) {
                     uint32_t delayedKvSIdx = kvSIdx - preLaunch * blockStackNum;
                     if (delayedKvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                         stackSeqTile = noMaskKvS - kvSIdx * pagedBlockSize;
                     } else {
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     LayoutO layoutO(qSeqlen, embed * qHeads);
                     LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                     GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                     uint32_t curStackTileMod = (stackSeqCount - preLaunch) % (preLaunch + 1);
                     uint32_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1)
                                             + curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                     Arch::CrossCoreWaitFlag(pvReady);
                     // rescale O
                     epilogueRescaleO(
                         gO[gmOffsetO], gOTmp[gmOffsetOTmp], gSharedO[gmOffsetO], layoutO, layoutOTmp, actualBlockShapePV, qSBlockSize,
                         qNBlockSize, (stackSeqCount - preLaunch == 0), 0, curStackTileMod
                     );
                 }
                 stackSeqCount++;
             }
             /*
              * for the secondary loop
              * while masked, it deals the CV stage1(Qk^t/SMOnline) of the final base block(typical shape [128, 512]),
              * and the CV stage2(PV/rescaleO) of the last (prelaunch+1) base blocks while unmasked, it deals the CV
              * stage1(Qk^t/SMOnline) of the last (prelaunch+1) base blocks
              */
             // deal secondary loop conditions
             uint32_t maskedStartIdx = (maskType != 0) ? ((noMaskTailS != 0) ? (kvSLoopNumNoMask - 1) : kvSLoopNumNoMask)
                                                       : AlignUp(kvSLoopNumNoMask, blockStackNum);
             uint32_t noMaskTailInteStackNum = (noMaskKvS / pagedBlockSize) % blockStackNum;
             noMaskTailInteStackNum = (noMaskTailInteStackNum != 0) ? noMaskTailInteStackNum
                                                                    : ((noMaskTailS != 0) ? 0 : blockStackNum);
             uint32_t preLaunchStackNum = (maskType != 0) ? ((preLaunch - 1) * blockStackNum + noMaskTailInteStackNum)
                                                          : (preLaunch * blockStackNum);
             // masked kvSeqlen loop
             // maskedStartIdx = AlignUp(kvSLoopNumNoMask, blockStackNum)
             // kvSLoopNumTotal = kvSLoopNumNoMask  so maskedStartIdx means kvSIdx >= kvSLoopNumTotal
             for (uint32_t kvSIdx = maskedStartIdx; kvSIdx < kvSLoopNumTotal + preLaunchStackNum;) {
                // In no mask scenario, kvSIdx will not be less than kvSLoopNumTotal, so it will not enter this if
                 if ((kvSIdx < kvSLoopNumTotal) && (stackSeqCount <= totalStackSeqNum - 1)) {
                    //  stackSeqTile = maskedKvS;
                    //  uint32_t stackSeqTileRound = RoundUp(stackSeqTile, BLOCK_SIZE);
                    //  LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
                    //  LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                    //  LayoutMask layOutMask(1024, 1024, 1024);
                    //  GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                    //  uint32_t curStackTileMod = stackSeqCount % (preLaunch + 1);
                    //  uint32_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + // cube core offset
                    //                       curStackTileMod * WORKSPACE_BLOCK_SIZE_DB; // single cube core db offset
                    //  // vec core offset will be processed within epilogue block
                    //  uint32_t gmOffsetP = gmOffsetS;
                    //  // online softmax
                    //  epilogueOnlineSoftmax(
                    //      gP[gmOffsetP], gS[gmOffsetS], gMask, layOutP, layOutS, layOutMask, actualBlockShapeQK,
                    //      (stackSeqCount == 0), qSBlockSize, qNBlockSize, curStackTileMod, qkReady
                    //  );
                     Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                 }
                 if (kvSIdx >= preLaunchStackNum) {
                     uint32_t delayedKvSIdx = kvSIdx - preLaunchStackNum;
                     if (delayedKvSIdx + blockStackNum > kvSLoopNumTotal - 1 && (maskType != 0)) {
                         stackSeqTile = maskedKvS;
                     } else if (delayedKvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                         stackSeqTile = noMaskKvS - delayedKvSIdx * pagedBlockSize;
                     } else {
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     LayoutO layoutO(qSBlockSize, embed * qHeads);
                     LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                     GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                     uint32_t curStackTileMod = (stackSeqCount - preLaunch) % (preLaunch + 1);
                     uint32_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1)
                                             + curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                     Arch::CrossCoreWaitFlag(pvReady);
                     // rescale O
                     epilogueRescaleO(
                         gO[gmOffsetO], gOTmp[gmOffsetOTmp], gSharedO[gmOffsetO], layoutO, layoutOTmp, actualBlockShapePV, qSBlockSize,
                         qNBlockSize, (stackSeqCount - preLaunch == 0),
                         (stackSeqCount - preLaunch == totalStackSeqNum - 1), curStackTileMod
                     );
                 }
                 if ((maskType != 0) && (stackSeqCount - preLaunch == totalStackSeqNum - 2)) {
                     kvSIdx += noMaskTailInteStackNum;
                 } else {
                     kvSIdx += blockStackNum;
                 }
                 stackSeqCount++;
             }
         }
 
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
 
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
     }
 
 private:
     Arch::Resource<ArchTag> resource;
     Arch::CrossCoreFlag qkReady{QK_READY_ID};
     Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
     Arch::CrossCoreFlag pvReady{PV_READY_ID};
     XAttentionTilingData* faTilingData;
 };
 

 /*
     FASharedInferKernelShort
     Compute Stream 短序列优化场景
     1. BlockMmadQK
     2. OnlineSoftmax
     3. BlockMmadPV
     4. EpilogueRescaleO
 */
 template <
     class BlockMmadQK,
     class BlockMmadPV,
     class EpilogueOnlineSoftmax,
     class EpilogueRescaleO,
     bool PAGED_CACHE_FLAG = true
 >
 class SharedFAInferKernelShort {
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
 
     using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
     using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;
     // Methods
     CATLASS_DEVICE
     SharedFAInferKernelShort(XAttentionTilingData* tilingDataPtr): faTilingData(tilingDataPtr) {
     }
 

    struct TaskQue {
        uint32_t taskIdx;
        uint64_t gmOffsetV;
        uint64_t gmOffsetO;
        uint32_t rowNum;
        uint32_t stackSeqTile;
        uint32_t kvSIdx;
        uint32_t blockBOffset;
        uint32_t qSeqlen;
        uint32_t qSBlockSize;
        uint32_t qNBlockSize;
        uint32_t kvSLoopNumTotal;

        CATLASS_DEVICE
        TaskQue()
        {}

        CATLASS_DEVICE
        void SetValue(uint32_t taskIdx_, uint64_t gmOffsetV_, uint64_t gmOffsetO_, uint32_t rowNum_,
            uint32_t stackSeqTile_, uint32_t kvSIdx_, uint32_t blockBOffset_, uint32_t qSeqlen_, uint32_t qSBlockSize_,
            uint32_t qNBlockSize_, uint32_t kvSLoopNumTotal_)
        {
            taskIdx = taskIdx_;
            gmOffsetV = gmOffsetV_;
            gmOffsetO = gmOffsetO_;
            rowNum = rowNum_;
            stackSeqTile = stackSeqTile_;
            kvSIdx = kvSIdx_;
            blockBOffset = blockBOffset_;
            qSeqlen = qSeqlen_;
            qSBlockSize = qSBlockSize_;
            qNBlockSize = qNBlockSize_;
            kvSLoopNumTotal = kvSLoopNumTotal_;
        }
    };

     template <int32_t CORE_TYPE = g_coreType>
     CATLASS_DEVICE void operator()(XAttnKernelParams const &params);
 
     template <>
     CATLASS_DEVICE void operator()<AscendC::AIC>(XAttnKernelParams const &params) {
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);

         AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        // TODO L1TileShape可能会有区别
        static constexpr uint32_t L1_QK_SIZE =
            BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K * sizeof(ElementQ);

         BlockMmadQK blockMmadQK(resource);
         BlockMmadPV blockMmadPV(resource, L1_QK_SIZE);
 
         //__gm__ XATilingData *faTilingData = reinterpret_cast<__gm__ XATilingData *>(params.tiling);
         uint64_t mm1OutSize = faTilingData->mm1OutSize;
         uint64_t smOnlineOutSize = faTilingData->smOnlineOutSize;
         uint32_t batch = faTilingData->batch;  // requestNum 
         uint32_t beamSize = faTilingData->beamSize; 
         uint32_t qHeads = faTilingData->numHeads;
         uint32_t kvHeads = faTilingData->kvHeads;
         uint32_t embed = faTilingData->embeddingSize;
         uint32_t pagedBlockSize = faTilingData->blockSize;
         uint32_t sharedCoreNum = faTilingData->sharedCoreNum;
         uint32_t maxNumBlocksPerBatch = faTilingData->maxNumBlocksPerBatch;
         uint32_t curTotalTaskNum = faTilingData->firstSharedBatchTaskNum;
         uint32_t totalTaskNum = faTilingData->sharedTotalTaskNum;
         uint32_t blockSize = faTilingData->blockSize;
         uint32_t maskType = faTilingData->maskType;
         float scaleValue = faTilingData->scaleValue;
 
         AscendC::GlobalTensor<ElementQ> gQ;
         gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
         AscendC::GlobalTensor<ElementK> gK;
         gK.SetGlobalBuffer((__gm__ ElementK *)params.k_cache);
         AscendC::GlobalTensor<ElementK> gV;
         gV.SetGlobalBuffer((__gm__ ElementK *)params.v_cache);
         AscendC::GlobalTensor<int32_t> gBlockTable;
         gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.sharedBlockTable));
         //AscendC::GlobalTensor<int64_t> gActualQseqlen;
         //gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
         AscendC::GlobalTensor<int32_t> gActualKvseqlen;
         gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualKvseqlen);
         AscendC::GlobalTensor<ElementS> gS;
         gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
         AscendC::GlobalTensor<ElementP> gP;
         gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
         AscendC::GlobalTensor<ElementOTmp> gOTmp;
         gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTemp);
         AscendC::GlobalTensor<ElementS> gSharedO;
         gSharedO.SetGlobalBuffer((__gm__ ElementS *)params.shared_workspace);
 
         uint64_t strideQO = qHeads * embed;
         uint64_t strideKV = kvHeads * embed;
         uint32_t embedRound = RoundUp<BLOCK_SIZE>(embed);
         uint32_t groupSize = qHeads / kvHeads;
 
         uint32_t coreIdx = AscendC::GetBlockIdx();
         uint32_t coreNum = sharedCoreNum; // TODO coreNum Need To be modified
         uint32_t preTotalTaskNum = 0;
         uint32_t curBatch = 0;
         uint64_t qBOffset = 0;
         uint64_t kBOffset = 0;
         uint64_t vBOffset = 0;
         uint64_t blockBOffset = 0;
         uint64_t oBOffset = 0; 

         int64_t qSeqlen = 0;
         int64_t kvSeqlen = 0;
         uint32_t curQNBlockTile;
         uint32_t qNBlockNumPerGroup;
         uint32_t curQNBlockNum;
         int64_t curQSBlockTile;
         uint32_t curQSBlockNum;
         uint32_t blockStackNum = 4;
         uint32_t stackSeqTilePad = blockStackNum * pagedBlockSize;

         qSeqlen = beamSize;
         kvSeqlen = gActualKvseqlen.GetValue(curBatch);
         curQSBlockTile = GetQSBlockTile(kvSeqlen);
         curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
         qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
         curQNBlockNum = qNBlockNumPerGroup * kvHeads;
         curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
        
         uint32_t l1KPPingPongFlag = 0;
         uint32_t l0ABPingPongFlag = 0;
         uint32_t l0CPingPongFlag = 0;
         uint32_t workspaceStagesFlag = 0;
         uint32_t loopCnt = 0;
         TaskQue taskQue[PRE_LAUNCH + 1];
         LayoutK layoutKTemp(strideKV, stackSeqTilePad);
         LayoutV layoutVTemp(stackSeqTilePad, strideKV);

         for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum + PRE_LAUNCH * coreNum; taskIdx += uint32_t(coreNum)) {
            while(taskIdx >= curTotalTaskNum && taskIdx < totalTaskNum) {
                ++curBatch;
                preTotalTaskNum = curTotalTaskNum;
                qBOffset += qSeqlen * strideQO;
                if (!PAGED_CACHE_FLAG) {
                    kBOffset += kvSeqlen * strideKV;
                    vBOffset += kvSeqlen * strideKV;
                } else {
                    blockBOffset += maxNumBlocksPerBatch;
                }
                oBOffset += qSeqlen * strideQO;

                kvSeqlen = gActualKvseqlen.GetValue(curBatch);
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

            uint32_t kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
            uint32_t qHeadIdx = kvHeadIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;

            uint64_t gmOffsetQ = qBOffset + qSBlockIdx * curQSBlockTile * strideQO + qHeadIdx * embed;
            uint64_t gmOffsetK = kBOffset + kvHeadIdx * embed;
            uint64_t gmOffsetV = vBOffset + kvHeadIdx * embed;
            uint64_t gmOffsetO = oBOffset + qSBlockIdx * curQSBlockTile * strideQO + qHeadIdx * embed;

            uint32_t qSBlockSize =
                (qSBlockIdx == (curQSBlockNum - 1)) ? (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
            uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1))
                                       ? (groupSize - qNBlockIdxCurGroup * curQNBlockTile)
                                       : curQNBlockTile; 

            uint32_t rowNum = qSBlockSize * qNBlockSize;
            uint32_t noSkipKvS = kvSeqlen;
            uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, pagedBlockSize);
            if (taskIdx >= totalTaskNum) {
                kvSLoopNumTotal = 1;
            }

            uint32_t stackSeqTile;

            LayoutQ layoutQTemp(rowNum, embed);
            if (taskIdx < totalTaskNum) {
                blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qHeads);
            } 

            for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal; kvSIdx += blockStackNum) {
                if (taskIdx < totalTaskNum) {
                    if (kvSIdx + blockStackNum > kvSLoopNumTotal - 1) {
                        stackSeqTile = noSkipKvS - kvSIdx * pagedBlockSize;
                    } else {
                        stackSeqTile = stackSeqTilePad;
                    }
                    uint64_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1)
                    + workspaceStagesFlag * WORKSPACE_BLOCK_SIZE_DB;
                    uint32_t taskStagesFlag = (taskIdx / coreNum) % (PRE_LAUNCH + 1);
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                    LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);

                    blockMmadQK(gQ[gmOffsetQ],
                    gK[gmOffsetK],
                    gS[gmOffsetS],
                    gBlockTable[blockBOffset],
                    layoutQTemp,
                    layoutKTemp,
                    layOutS,
                    actualBlockShapeQK,
                    kvSIdx,
                    pagedBlockSize,
                    strideKV,
                    l1KPPingPongFlag,
                    l0ABPingPongFlag,
                    l0CPingPongFlag);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);

                    taskQue[workspaceStagesFlag].SetValue(taskIdx,
                        gmOffsetV,
                        gmOffsetO,
                        rowNum,
                        stackSeqTile,
                        kvSIdx,
                        blockBOffset,
                        qSeqlen,
                        qSBlockSize,
                        qNBlockSize,
                        kvSLoopNumTotal);
                }
                if (loopCnt >= PRE_LAUNCH) {
                    uint32_t nowWorkspaceStagesFlag =
                        (workspaceStagesFlag + (PRE_LAUNCH + 1) - PRE_LAUNCH) % (PRE_LAUNCH + 1);
                    uint32_t nowTaskStagesFlag = (taskQue[nowWorkspaceStagesFlag].taskIdx / coreNum) % (PRE_LAUNCH + 1);
                    uint32_t nowkvSIdx = taskQue[nowWorkspaceStagesFlag].kvSIdx;
                    uint32_t nowRowNum = taskQue[nowWorkspaceStagesFlag].rowNum;
                    stackSeqTile = taskQue[nowWorkspaceStagesFlag].stackSeqTile;
                    uint64_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                            nowWorkspaceStagesFlag * WORKSPACE_BLOCK_SIZE_DB;
                    GemmCoord actualBlockShapePV{nowRowNum, embed, stackSeqTile};
                    LayoutOTmp layoutOTmp(nowRowNum, embed, embedRound);
                    LayoutP layoutPTemp(nowRowNum, stackSeqTile, stackSeqTilePad);
                    uint64_t gmOffsetP = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                         nowWorkspaceStagesFlag * WORKSPACE_BLOCK_SIZE_DB;
                    blockMmadPV(gP[gmOffsetP],
                        gV[taskQue[nowWorkspaceStagesFlag].gmOffsetV],
                        gOTmp[gmOffsetOTmp],
                        gBlockTable[taskQue[nowWorkspaceStagesFlag].blockBOffset],
                        layoutPTemp,
                        layoutVTemp,
                        layoutOTmp,
                        actualBlockShapePV,
                        nowkvSIdx,
                        pagedBlockSize,
                        strideKV,
                        softmaxReady,
                        l1KPPingPongFlag,
                        l0ABPingPongFlag,
                        l0CPingPongFlag);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                }
                loopCnt++;
                workspaceStagesFlag = (workspaceStagesFlag + 1) % (PRE_LAUNCH + 1);
            }

         }
 
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
 
         AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
 
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
     }
 
 
     template <>
     CATLASS_DEVICE void operator()<AscendC::AIV>(XAttnKernelParams const &params) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
 
         // Get tiling parameters
         //__gm__ XATilingData *faTilingData = reinterpret_cast<__gm__ XATilingData *>(params.tiling);
         uint64_t mm1OutSize = faTilingData->mm1OutSize;
         uint64_t smOnlineOutSize = faTilingData->smOnlineOutSize;
         uint64_t mm2OutSize = faTilingData->mm2OutSize;
         uint32_t batch = faTilingData->batch;
         uint32_t beamSize = faTilingData->beamSize;
         uint32_t qHeads = faTilingData->numHeads;
         uint32_t kvHeads = faTilingData->kvHeads;
         uint32_t embed = faTilingData->embeddingSize;
         uint32_t sharedCoreNum = faTilingData->sharedCoreNum;
         uint32_t pagedBlockSize = faTilingData->blockSize;
         uint32_t maxNumBlocksPerBatch = faTilingData->maxNumBlocksPerBatch;
         uint32_t firstBatchTaskNum = faTilingData->firstSharedBatchTaskNum;
         uint32_t totalTaskNum = faTilingData->sharedTotalTaskNum;
         uint32_t maskType = faTilingData->maskType;
         float scaleValue = faTilingData->scaleValue;
         
         uint64_t gOffsetTempO = batch * beamSize * qHeads * embed;
         uint64_t gMaxOffset = batch * beamSize * qHeads * SOFTMAX_BROAD_SIZE;
         // Get the memory offset address of the input on Global Memory
         //AscendC::GlobalTensor<ElementMask> gMask;
         //gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
         //AscendC::GlobalTensor<int64_t> gActualQseqlen;
         //gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
         AscendC::GlobalTensor<int32_t> gActualKvseqlen;
         gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualKvseqlen);
         AscendC::GlobalTensor<ElementO> gO;
         gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
         AscendC::GlobalTensor<ElementS> gS;
         gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
         AscendC::GlobalTensor<ElementP> gP;
         gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
         AscendC::GlobalTensor<ElementOTmp> gOTmp;
         gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTemp);
         AscendC::GlobalTensor<ElementOTmp> gOUpdate;
         gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)params.oUpdate);
 
         // shared Gm and Gl output
         AscendC::GlobalTensor<float> gSharedO;
         AscendC::GlobalTensor<ElementS> gSharedSum;
         AscendC::GlobalTensor<ElementS> gSharedMax;
         gSharedO.SetGlobalBuffer((__gm__ float *)params.shared_workspace);
         gSharedMax.SetGlobalBuffer((__gm__ ElementS *)params.shared_workspace + gOffsetTempO);
         gSharedSum.SetGlobalBuffer((__gm__ ElementS *)params.shared_workspace + gOffsetTempO + gMaxOffset);
         

         uint32_t groupSize = qHeads / kvHeads;
         uint32_t embedRound = RoundUp(embed, BLOCK_SIZE);
 
         EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
         EpilogueRescaleO epilogueRescaleO(resource);
 
         // uint32_t curTotalTaskNum = 0;
         uint32_t preTotalTaskNum = 0;
         uint32_t curBatch = 0;
         uint32_t oBOffset = 0;
         uint32_t qSeqlen = static_cast<uint32_t>(beamSize);
         uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
         uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
         uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
         uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;
         uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
         uint32_t curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
         uint32_t curTotalTaskNum = firstBatchTaskNum;
         uint32_t blockStackNum = 4;
         uint32_t stackSeqTilePad = blockStackNum * pagedBlockSize; 
         uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
         // coreNum Need to be changed TODO
         uint32_t coreNum = sharedCoreNum;

         TaskQue taskQue[PRE_LAUNCH + 1];
         uint32_t loopCnt = 0;
         uint32_t workspaceStagesFlag = 0;
         // Go through each task.
         for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum + PRE_LAUNCH * coreNum; taskIdx += uint32_t(coreNum)) {
             // Get the offset of each core on the GM.
             while (taskIdx >= curTotalTaskNum && taskIdx < totalTaskNum) {
                 curBatch++;
                 oBOffset += qSeqlen * qHeads * embed;
                 preTotalTaskNum = curTotalTaskNum;
                 qSeqlen = static_cast<uint32_t>(beamSize);
                 kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                 curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                 qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                 curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                 curQSBlockTile = GetQSBlockTile(kvSeqlen);
                 curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
                 curTotalTaskNum += curQNBlockNum * curQSBlockNum;
             }
             uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
             uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
             uint32_t qNBlockIdx = taskIdxCurBatch % curQNBlockNum;
             uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
 
             uint32_t oSOffset = qSBlockIdx * curQSBlockTile * qHeads * embed;
             uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
             uint32_t qStartNIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
             uint32_t oNOffset = qStartNIdx * embed;
             uint32_t gmOffsetO = oBOffset + oSOffset + oNOffset;
    
             // shared sum max workspace offset
             // Calculate shared workspace offset for this core's task
             // gsharedout size: [batch*beamSize, numHeads, 8]
             // Each core handles: [qSBlockSize, qNBlockSize] elements
             // Offset = curBatch * beamSize * qHeads * 8 + qSBlockIdx * curQSBlockTile * qHeads * 8 + qStartNIdx * 8;
             uint32_t gSharedOffset = curBatch * qSeqlen * qHeads * SOFTMAX_BROAD_SIZE + 
             qSBlockIdx * curQSBlockTile * qHeads * SOFTMAX_BROAD_SIZE + qStartNIdx * SOFTMAX_BROAD_SIZE;
             
             
             uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1)) ? (qSeqlen - qSBlockIdx * curQSBlockTile)
                                                                        : curQSBlockTile;
             uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1))
                                        ? (groupSize - qNBlockIdxCurGroup * curQNBlockTile)
                                        : curQNBlockTile;
             uint32_t rowNum = qSBlockSize * qNBlockSize;
             uint32_t rowNumRound = RoundUp(rowNum, BLOCK_SIZE);
 
             uint32_t noSkipKvS = kvSeqlen;
             uint32_t noMaskKvS = kvSeqlen;
             uint32_t noMaskTailS = 0;
            
             uint32_t stackSeqTile = 0;
             uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, pagedBlockSize);
             if (taskIdx >= totalTaskNum) {
                kvSLoopNumTotal = 1;
             }
             uint32_t isLastStackTile = 0;
             // no mask kvSeqlen loop
             for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal; kvSIdx += blockStackNum) {
                if (taskIdx < totalTaskNum) {
                    if (kvSIdx + blockStackNum > kvSLoopNumTotal - 1) {
                        stackSeqTile = noSkipKvS - kvSIdx * pagedBlockSize;
                        isLastStackTile = 1;
                    } else {
                        stackSeqTile = stackSeqTilePad;
                    }
                    uint64_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                         workspaceStagesFlag * WORKSPACE_BLOCK_SIZE_DB;
                    uint32_t taskStagesFlag = (taskIdx / coreNum) % (PRE_LAUNCH + 1);
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                    LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad); 
                    LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                    uint64_t gmOffsetP = gmOffsetS;
                    uint32_t kvSStartIdx = kvSIdx * pagedBlockSize;
                    uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;
                    Arch::CrossCoreWaitFlag(qkReady);
                    epilogueOnlineSoftmax(
                        gP[gmOffsetP], gS[gmOffsetS], gSharedMax[gSharedOffset], gSharedSum[gSharedOffset], 
                        layOutP, layOutS, actualBlockShapeQK, (kvSIdx == 0),
                        isLastStackTile, qSBlockSize, qNBlockSize, workspaceStagesFlag, qHeads
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                    taskQue[workspaceStagesFlag].SetValue(taskIdx,
                    0,
                    gmOffsetO,
                    rowNum,
                    stackSeqTile,
                    kvSIdx,
                    0,
                    qSeqlen,
                    qSBlockSize,
                    qNBlockSize,
                    kvSLoopNumTotal);

                }
                if (loopCnt >= PRE_LAUNCH) {
                    uint32_t nowWorkspaceStagesFlag =
                        (workspaceStagesFlag + (PRE_LAUNCH + 1) - PRE_LAUNCH) % (PRE_LAUNCH + 1);
                    uint32_t nowTaskStagesFlag = (taskQue[nowWorkspaceStagesFlag].taskIdx / coreNum) % (PRE_LAUNCH + 1);
                    uint32_t nowkvSIdx = taskQue[nowWorkspaceStagesFlag].kvSIdx;
                    uint32_t nowRowNum = taskQue[nowWorkspaceStagesFlag].rowNum;
                    stackSeqTile = taskQue[nowWorkspaceStagesFlag].stackSeqTile;
                    // uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1);
                    uint64_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                            nowWorkspaceStagesFlag * WORKSPACE_BLOCK_SIZE_DB;
                    GemmCoord actualBlockShapePV{nowRowNum, embed, stackSeqTile};
                    LayoutOTmp layoutOTmp(nowRowNum, embed, embedRound);
                                        LayoutO layoutO(taskQue[nowWorkspaceStagesFlag].qSeqlen, embed * qHeads);
                    LayoutUpdate layoutUpdate(nowRowNum, embed, embedRound);
                    uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * WORKSPACE_BLOCK_SIZE_DB);

                    Arch::CrossCoreWaitFlag(pvReady);
                    // rescale O
                    epilogueRescaleO(
                        gO[taskQue[nowWorkspaceStagesFlag].gmOffsetO], 
                        gOTmp[gmOffsetOTmp], 
                        gSharedO[taskQue[nowWorkspaceStagesFlag].gmOffsetO], 
                        layoutO, layoutOTmp, actualBlockShapePV, 
                        taskQue[nowWorkspaceStagesFlag].qSBlockSize,
                        taskQue[nowWorkspaceStagesFlag].qNBlockSize,
                        (nowkvSIdx == 0),
                        nowkvSIdx + blockStackNum >= taskQue[nowWorkspaceStagesFlag].kvSLoopNumTotal, 
                        nowWorkspaceStagesFlag
                    );
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                }
                loopCnt++;
                workspaceStagesFlag = (workspaceStagesFlag + 1) % (PRE_LAUNCH + 1); 
                     
             }
         }
 
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
     }
 
 private:
     Arch::Resource<ArchTag> resource;
     Arch::CrossCoreFlag qkReady{QK_READY_ID};
     Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
     Arch::CrossCoreFlag pvReady{PV_READY_ID};
     XAttentionTilingData* faTilingData;
 };


template <class EpilogueCombineScale>
class CombineScaleKernel {
public:
    using ArchTag = typename EpilogueCombineScale::ArchTag;
    using ElementOutput = typename EpilogueCombineScale::ElementOutput;
    using ElementInput = typename EpilogueCombineScale::ElementInput;

    CATLASS_DEVICE
    CombineScaleKernel(XAttentionTilingData* tilingDataPtr): faTilingData(tilingDataPtr) {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(XAttnKernelParams const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(XAttnKernelParams const &params) {
        return;
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(XAttnKernelParams const &params) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

        EpilogueCombineScale epilogueCombineScale(resource);

        // get tiling params
        // __gm__ XATilingData *faTilingData = reinterpret_cast<__gm__ XATilingData *>(params.tiling);
        uint32_t combineFormerCoreNum = faTilingData->combineFormerCoreNum;
        uint32_t combineFormerRowNum = faTilingData->combineFormerRowNum;
        uint32_t combineTailRowNum = faTilingData->combineTailRowNum;
        uint32_t numTokens = faTilingData->numTokens;
        uint32_t qHeads = faTilingData->numHeads;
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t currentCoreRowNum = 0;
        uint32_t embed = faTilingData->embeddingSize;
        uint32_t gmOffsetO = 0;
        uint32_t gmOffsetSharedGl = 0;
        uint32_t gmOffsetSharedGm = 0;
        uint32_t gmOffsetUnSharedGl = 0;
        uint32_t gmOffsetUnSharedGm = 0;
        uint32_t sharedSumMaxGmOffset = numTokens * qHeads * SOFTMAX_BROAD_SIZE;
        uint32_t unsharedSumMaxGmOffset = numTokens * qHeads;
        uint32_t attnOutputOffset = numTokens * qHeads * embed;
        if (coreIdx >= faTilingData->combineCoreNum) {
            return;
        }

        if (coreIdx < combineFormerCoreNum) {
            currentCoreRowNum = combineFormerRowNum;
            gmOffsetO = coreIdx * combineFormerRowNum * embed;
            gmOffsetUnSharedGl = coreIdx *combineFormerRowNum;
            gmOffsetUnSharedGm = gmOffsetUnSharedGl;
            gmOffsetSharedGl = coreIdx *combineFormerRowNum * SOFTMAX_BROAD_SIZE;
            gmOffsetSharedGm = gmOffsetSharedGl;
        } else {
            currentCoreRowNum = combineTailRowNum;
            gmOffsetO = combineFormerCoreNum * combineFormerRowNum * embed 
            + (coreIdx - combineFormerCoreNum) * combineTailRowNum * embed;
            gmOffsetUnSharedGl = combineFormerCoreNum * combineFormerRowNum
            + (coreIdx - combineFormerCoreNum) * combineTailRowNum;
            gmOffsetUnSharedGm = gmOffsetUnSharedGl;
            gmOffsetSharedGl = gmOffsetUnSharedGl * SOFTMAX_BROAD_SIZE;
            gmOffsetSharedGm = gmOffsetSharedGl;
        }
        // shared_workspace [attnout:[attnOut * 4Bytes], 
        // gm:[attnOutputOffset * 4Bytes], 
        // gl:[attnOutputOffset * 4Bytes]]
        AscendC::GlobalTensor<ElementInput> gSharedGm;
        gSharedGm.SetGlobalBuffer((__gm__ ElementInput *)params.shared_workspace + attnOutputOffset);
        AscendC::GlobalTensor<ElementInput> gSharedGl;
        gSharedGl.SetGlobalBuffer((__gm__ ElementInput *)params.shared_workspace + attnOutputOffset +
                                  sharedSumMaxGmOffset);
        AscendC::GlobalTensor<ElementInput> gUnsharedGm;
        gUnsharedGm.SetGlobalBuffer((__gm__ ElementInput *)params.unshared_workspace + attnOutputOffset);
        AscendC::GlobalTensor<ElementInput> gUnsharedGl;
        gUnsharedGl.SetGlobalBuffer((__gm__ ElementInput *)params.unshared_workspace + attnOutputOffset +
                                    unsharedSumMaxGmOffset);
        AscendC::GlobalTensor<ElementInput> gSharedOut;
        gSharedOut.SetGlobalBuffer((__gm__ ElementInput *)params.shared_workspace);
        AscendC::GlobalTensor<ElementInput> gUnsharedOut;
        gUnsharedOut.SetGlobalBuffer((__gm__ ElementInput *)params.unshared_workspace);
        AscendC::GlobalTensor<ElementOutput> gFinalOut;
        gFinalOut.SetGlobalBuffer((__gm__ ElementOutput *)params.o);

        MatrixCoord actualBlockShape(currentCoreRowNum, embed);
        //cce::printf("coreIdx:%d, gmOffsetO:%d, currentCoreRowNum:%d, gmOffsetGl:%d, gmOffsetGm:%d\n", AscendC::GetBlockIdx(), 
        //gmOffsetO, currentCoreRowNum, gmOffsetGl, gmOffsetGm);
        epilogueCombineScale(
            gSharedGm[gmOffsetSharedGm], gUnsharedGm[gmOffsetUnSharedGm],
            gSharedGl[gmOffsetSharedGl], gUnsharedGl[gmOffsetUnSharedGl],
            gSharedOut[gmOffsetO], gUnsharedOut[gmOffsetO], 
            gFinalOut[gmOffsetO], actualBlockShape
        );
        
        // AscendC::PipeBarrier<PIPE_ALL>();
        
        
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
    }
private:
    Arch::Resource<ArchTag> resource;
    XAttentionTilingData* faTilingData;
};

#endif