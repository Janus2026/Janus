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
using namespace AscendC;

class KernelMatmulInt128Mlt512 {
public:

  __aicore__ inline KernelMatmulInt128Mlt512() {}
  
  // Main interface functions
  __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale,
       __gm__ uint8_t *perTokenScale, __gm__ uint8_t *groupoffset,
       __gm__ uint8_t *out, __gm__ uint8_t *workspace, uint32_t M, uint32_t N,
       uint32_t K, uint32_t baseM, uint32_t baseN, uint32_t baseK,
       uint32_t tailM, uint32_t tailN, uint32_t tailK, uint32_t groupNum,
       uint32_t actExperts);
  
  __aicore__ inline void Process();

private:
  // ==================== Initialization related functions ====================
  __aicore__ inline void initVector(uint32_t splitValue, uint32_t offsetN);
  __aicore__ inline void DataPreload(uint64_t groupIdx, uint32_t preoffset);
  
  // ==================== Dequantization correlation function ====================
  __aicore__ inline void Dequant(uint64_t mmOutOffset, uint32_t splitValue,
                                 uint32_t groupIdx, uint32_t offsetN,
                                 uint32_t vectorBaseN, uint32_t offsetNMax,
                                 uint32_t singleMId);
  __aicore__ inline void CopyPerTokenScale(uint64_t mmOutOffset,
                                           uint32_t splitValue,
                                           uint32_t singleMId);
  __aicore__ inline void CopycGm(uint32_t mmOutOffset, uint32_t offsetN,
                                 uint32_t splitValue, uint32_t singleMId);
  __aicore__ inline void CopyScale(uint32_t groupIdx, uint32_t offsetN);
  
  // ==================== Matrix multiplication calculation related functions ====================
  __aicore__ inline void MMCompute(uint32_t groupIdx, uint32_t splitValue,
                                   int32_t *tokensIdx, uint32_t oneTimesData,
                                   uint32_t preOffset);
  __aicore__ inline void CopyA(uint32_t splitValue, uint32_t preOffset,
                               int32_t *tokensIdx, uint32_t singleMId);
  __aicore__ inline void CopyIn(uint32_t singleMId, uint32_t tilingNumId,
                                uint32_t singleNId, uint32_t groupIdx,
                                uint32_t splitValue, int32_t *tokensIdx,
                                uint32_t oneTimesData, uint32_t numbers);
  __aicore__ inline void CopyOut(uint32_t singleMId, uint32_t singleNId,
                                 uint32_t preOffset, uint32_t oneTimesData);
  
  __aicore__ inline void SplitA(uint32_t singleMId, uint32_t singleKId,
                                uint32_t singleNId);
  __aicore__ inline void SplitB(uint32_t singleMId, uint32_t singleKId,
                                uint32_t singleNId);
  __aicore__ inline void Compute(uint32_t singleMId, uint32_t singleKId,
                                 uint32_t singleNId,
                                 AscendC::LocalTensor<int32_t> c1Local);
  
  // ==================== Utility Functions ====================
  __aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t base);
  template <typename T>
  __aicore__ inline __gm__ T* GetTensorAddr(GM_ADDR tensorPtr);

private:
  // ==================== Pipeline and queue related members ====================
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::A1, 1> inQueueA1;
  AscendC::TQue<AscendC::QuePosition::A2, 1> inQueueA2;
  AscendC::TQue<AscendC::QuePosition::B1, 1> inQueueB1;
  AscendC::TQue<AscendC::QuePosition::B2, 1> inQueueB2;
  AscendC::TQue<AscendC::QuePosition::CO1, 1> outQueueCO1;
  AscendC::TQue<AscendC::QuePosition::C1, 1> inQueueC1;
  AscendC::TQue<AscendC::QuePosition::C2, 1> outQueueC2;
  AscendC::TBuf<TPosition::VECOUT> tmpBuff;
  
  // ==================== Global memory tensors ====================
  AscendC::GlobalTensor<int8_t> aGM;
  AscendC::GlobalTensor<int8_t> bGM;
  AscendC::GlobalTensor<int32_t> cGM;
  AscendC::GlobalTensor<int32_t> biasGM;
  AscendC::GlobalTensor<int64_t> groupoffsetGM;
  AscendC::GlobalTensor<bfloat16_t> outGM;
  AscendC::GlobalTensor<bfloat16_t> scaleGM;
  AscendC::GlobalTensor<float> perTokenScaleGM;
  
  // ==================== Local Tensors ====================
  AscendC::LocalTensor<int8_t> aLocal;
  AscendC::LocalTensor<int8_t> bLocal;
  
  // ==================== Quantify related queues and buffers ====================
  TQue<QuePosition::VECIN, 1> vecInQueue;
  TQue<QuePosition::VECOUT, 1> vecOutQueue;
  TQue<QuePosition::VECIN, 1> scaleInQueue;
  TQue<QuePosition::VECIN, 1> perTokenScaleInQueue;
  TBuf<TPosition::VECCALC> tmpBuffQuant;
  
  // ==================== Quantized calculation of related local tensors ====================
  LocalTensor<int32_t> mmOutInUb;
  LocalTensor<bfloat16_t> scaleInUb;
  LocalTensor<float> perTokenScaleInUb;
  LocalTensor<float> pertokenBrcbLocal;
  LocalTensor<float> dequantMiddleResult;
  LocalTensor<uint8_t> sharedTmpLocal;
  LocalTensor<float> mulsResultLocal;
  LocalTensor<float> actResultLocal;
  
  // ==================== Matrix dimension parameters ====================
  uint32_t M, N, K;
  uint32_t baseM, baseK, baseN;
  uint32_t tailM = 0, tailN = 0, tailK = 0;
  uint32_t aSize, bSize, cSize, sortedListSize;
  
  // ==================== Matrix dimension parameters ====================
  uint32_t roundM = 1; // M number of cutting cycles
  uint32_t roundK = 1; // K number of block cutting cycles
  uint32_t roundN = 1; // N-cut cycle
  uint32_t mblocks = 1;
  uint32_t nblocks = 1;
  uint32_t kblocks = 1;
  
  uint32_t groupNum;
  uint32_t actExperts;
  int32_t *sortedList;
  
  // ==================== Quantization related parameters ====================
  uint32_t scaleSize, perTokenScaleSize;
  bool isPerTokenQuant;
  bool sequentialWrite = true;
  
  // ==================== Core and synchronization related parameters ====================
  uint32_t subBlockIdx;
  uint32_t coreIdx;
  uint32_t cubeNum;
  uint32_t cubeTotalNums;
  uint64_t SYNC_AIC_AIV_FLAG = 1;
  uint64_t SYNC_AIV_AIC_FLAG = 1;
  int32_t PIPELINE_NUM = 1;
};

// ==================== Function implementation part ====================

__aicore__ inline void KernelMatmulInt128Mlt512::Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale,
       __gm__ uint8_t *perTokenScale, __gm__ uint8_t *groupoffset,
       __gm__ uint8_t *out, __gm__ uint8_t *workspace, uint32_t M, uint32_t N,
       uint32_t K, uint32_t baseM, uint32_t baseN, uint32_t baseK,
       uint32_t tailM, uint32_t tailN, uint32_t tailK, uint32_t groupNum,
       uint32_t actExperts) {
    // aGM.SetGlobalBuffer((__gm__ int8_t *)a);
    // bGM.SetGlobalBuffer((__gm__ int8_t *)b);
    cGM.SetGlobalBuffer((__gm__ int32_t *)workspace);
    // outGM.SetGlobalBuffer((__gm__ bfloat16_t *)out);
    aGM.SetGlobalBuffer(GetTensorAddr<int8_t>(a));
    bGM.SetGlobalBuffer(GetTensorAddr<int8_t>(b));
    //cGM.SetGlobalBuffer(GetTensorAddr<int32_t>(workspace));
    outGM.SetGlobalBuffer(GetTensorAddr<bfloat16_t>(out));
    groupoffsetGM.SetGlobalBuffer((__gm__ int64_t *)groupoffset);
    // if outGM is bfloat16_t,then scale use bfloat16_t, otherwise use float
    // scaleGM.SetGlobalBuffer((__gm__ bfloat16_t *)scale);
    // perTokenScaleGM.SetGlobalBuffer((__gm__ float *)perTokenScale);
    scaleGM.SetGlobalBuffer(GetTensorAddr<bfloat16_t>(scale));
    perTokenScaleGM.SetGlobalBuffer(GetTensorAddr<float>(perTokenScale));
    //  biasGM.SetGlobalBuffer((__gm__ int32_t *)bias);
    this->M = M;
    this->N = N;
    this->K = K;
    this->baseM = baseM;
    this->baseK = baseK;
    this->baseN = baseN;
    this->tailM = tailM;
    this->tailN = tailN;
    this->tailK = tailK;
    this->aSize = baseM * baseK;
    this->bSize = baseK * baseN;
    this->cSize = baseM * baseN;
    //  this->sortedListSize = M*actExperts;
    this->groupNum = groupNum;
    this->mblocks = baseM / 16;
    this->nblocks = baseN / 16;
    this->kblocks = baseK / 32;
    this->actExperts = actExperts;
    this->roundK = this->tailK == 0 ? K / baseK : K / baseK + 1;
    this->roundN = this->tailN == 0 ? N / baseN : N / baseN + 1;

    // TQue Initialization
    pipe.InitBuffer(inQueueA2, 2, aSize * sizeof(int8_t));
    pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(int8_t));
    pipe.InitBuffer(inQueueC1, 1, baseN * sizeof(int32_t));
    pipe.InitBuffer(outQueueC2, 1, baseN * sizeof(int32_t));
    // Get the number of all cores on the current device
    this->cubeTotalNums = AscendC::GetBlockNum();
    // Get the number of child cores of the current vector core and its
    // corresponding cube core
    this->subBlockIdx = AscendC::GetSubBlockIdx();
    // Get the current core number
    this->coreIdx = AscendC::GetBlockIdx();
    // vector inint
    if ASCEND_IS_AIV {
      this->coreIdx /= AscendC::GetTaskRation();
      initVector(baseM, baseN);
    }
}

__aicore__ inline void KernelMatmulInt128Mlt512::Process() {
    // preOffset is used to refer to the token from which each expert starts
    // calculating
    uint32_t preOffset = 0;
    // oneTimesData is used to indicate the number of data blocks moved from L1 to L0
    uint32_t oneTimesData = this->roundK > 2 ? 4 : 1;
    uint32_t curBlock = 0;
    // tokensIdx is the index of the token processed by the current expert in the x matrix
    int32_t tokensIdx[1024];
    pipe.InitBuffer(inQueueA1, 2, baseM * K * sizeof(int8_t));
    pipe.InitBuffer(inQueueB1, 2, oneTimesData * bSize * sizeof(int8_t));
    pipe.InitBuffer(outQueueCO1, 2, cSize * sizeof(int32_t));
    uint32_t splitValue;
    uint32_t num=0;
    for (size_t groupIdx = 0; groupIdx < this->groupNum; groupIdx++) {
      // Get how many tokens the current expert processes
      splitValue = groupoffsetGM.GetValue(groupIdx) - preOffset;
      if (splitValue == 0){
        continue;
      }
      // Assign groups to corresponding cores
      this->tailM = splitValue % baseM;
      this->roundM = this->tailM == 0 ? splitValue / baseM : splitValue / baseM + 1;
      if (curBlock % cubeTotalNums == coreIdx) {
        if ASCEND_IS_AIC {
          // Since the left matrix is ​​smaller, the entire left matrix is
          // ​​moved from GM to L1 at one time, so that the right matrix can
          // reuse the left matrix
          MMCompute(groupIdx, splitValue, tokensIdx, oneTimesData, preOffset);
          // After the cube calculation is completed, the vector core is
          // notified to start quantization
          CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
          //  inQueueA1.FreeTensor(aLocal);
        }
        if ASCEND_IS_AIV {
          // Waiting for cube to send signal
          CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);

          for (size_t singleMId = 0; singleMId < this->roundM; singleMId++) {
            bool isTail = (this->tailM != 0 && singleMId == this->roundM - 1);
            uint32_t currentSize = isTail ? this->tailM : baseM;
            CopyPerTokenScale(preOffset, currentSize, singleMId);
            uint32_t vectorRoundN = N / (baseN);
            int nNumbers = this->subBlockIdx == 0 ? 0 : vectorRoundN / 2;
            int offsetNMax = nNumbers + vectorRoundN / 2;
            CopycGm(preOffset, nNumbers, currentSize, singleMId);
            CopyScale(groupIdx, nNumbers);
            for (size_t offsetN = nNumbers; offsetN < offsetNMax; offsetN++) {
              Dequant(preOffset, currentSize, groupIdx, offsetN, baseN, offsetNMax, singleMId);
            }
            perTokenScaleInQueue.FreeTensor(perTokenScaleInUb);
          }
        }
      }
      curBlock++;
      preOffset += splitValue;
    }
}

// ==================== Initialization related function implementation ====================

__aicore__ inline void KernelMatmulInt128Mlt512::initVector(uint32_t splitValue, uint32_t offsetN) {
    // init
    this->scaleSize = offsetN;
    this->perTokenScaleSize = splitValue;

    pipe.InitBuffer(scaleInQueue, 2, offsetN * sizeof(bfloat16_t));
    pipe.InitBuffer(perTokenScaleInQueue, 2, offsetN * sizeof(float));
    pipe.InitBuffer(vecInQueue, 2, splitValue * offsetN * sizeof(int32_t));
    pipe.InitBuffer(vecOutQueue, 2, splitValue * offsetN * sizeof(bfloat16_t));
    pipe.InitBuffer(tmpBuffQuant, 6 * baseK * baseN * sizeof(uint8_t));
    int32_t tmpBuffQuantOffset = 0;
    uint32_t offset = splitValue * offsetN * sizeof(float);
    uint32_t offsetScale = splitValue * sizeof(float);
    dequantMiddleResult = tmpBuffQuant.GetWithOffset<float>(offset, tmpBuffQuantOffset);
    tmpBuffQuantOffset += offset;

    sharedTmpLocal = tmpBuffQuant.GetWithOffset<uint8_t>(offset, tmpBuffQuantOffset);
    tmpBuffQuantOffset += offset;

    pertokenBrcbLocal = tmpBuffQuant.GetWithOffset<float>(offset, tmpBuffQuantOffset);
    tmpBuffQuantOffset += offset;

    perTokenScaleInUb = tmpBuffQuant.GetWithOffset<float>(offsetScale, tmpBuffQuantOffset);
    tmpBuffQuantOffset += offsetScale;

    mulsResultLocal = tmpBuffQuant.GetWithOffset<float>(offset, tmpBuffQuantOffset);
    tmpBuffQuantOffset += offset;
}

__aicore__ inline void KernelMatmulInt128Mlt512::DataPreload(uint64_t groupIdx, uint32_t preoffset){
    LocalTensor<int8_t> dataLocal = tmpBuffQuant.Get<int8_t>();
    AscendC::DataCopy(dataLocal, aGM, 1*baseM*K);
    AscendC::DataCopy(dataLocal[1*baseM*K], bGM, 4*baseN*baseK);
}

// ==================== Implementation of inverse quantization correlation function ====================

__aicore__ inline void KernelMatmulInt128Mlt512::Dequant(uint64_t mmOutOffset, uint32_t splitValue,
                                 uint32_t groupIdx, uint32_t offsetN,
                                 uint32_t vectorBaseN, uint32_t offsetNMax,
                                 uint32_t singleMId) {
    //  // fisrt step:Move mmout to ub
    mmOutInUb = vecInQueue.DeQue<int32_t>();

    //  // step2:Move scale to ub
    scaleInUb = scaleInQueue.DeQue<bfloat16_t>();

    if (offsetN < offsetNMax - 1) {
      CopycGm(mmOutOffset, offsetN + 1, splitValue, singleMId);
      CopyScale(groupIdx, offsetN + 1);
    }
    // step3:Dequantize the output to float
    AscendDequant(dequantMiddleResult, mmOutInUb, scaleInUb, sharedTmpLocal,
                  {splitValue, vectorBaseN, vectorBaseN});
    scaleInQueue.FreeTensor(scaleInUb);
    vecInQueue.FreeTensor(mmOutInUb);
    uint32_t mulOffset = splitValue * vectorBaseN;
    Mul(mulsResultLocal, dequantMiddleResult, pertokenBrcbLocal, mulOffset);
    LocalTensor<bfloat16_t> yLocalInUb = vecOutQueue.AllocTensor<bfloat16_t>();
    Cast(yLocalInUb, mulsResultLocal, RoundMode::CAST_RINT, mulOffset);
    vecOutQueue.EnQue(yLocalInUb);

    //  step5:Move the quantified data from ub to Gm
    LocalTensor<bfloat16_t> yLocal = vecOutQueue.DeQue<bfloat16_t>();
    DataCopyParams outGmParams;
    outGmParams.blockCount = splitValue;
    outGmParams.blockLen = (vectorBaseN * sizeof(bfloat16_t)) / 32;
    outGmParams.srcStride = 0;
    outGmParams.dstStride =
        (this->N * sizeof(bfloat16_t) - vectorBaseN * sizeof(bfloat16_t)) / 32;
    uint32_t outGMOffset =
        (mmOutOffset + singleMId * baseM) * N + offsetN * vectorBaseN;
    AscendC::DataCopy(outGM[outGMOffset], yLocal, outGmParams);

    vecOutQueue.FreeTensor(yLocal);
}

__aicore__ inline void KernelMatmulInt128Mlt512::CopyPerTokenScale(uint64_t mmOutOffset,
                                           uint32_t splitValue,
                                           uint32_t singleMId) {
    // step4:Quantize float to bf16
    DataCopyPadExtParams<float> padParams;
    padParams.isPad = true;
    padParams.paddingValue = 0;
    DataCopyExtParams perTokenScaleParams;
    perTokenScaleParams.blockLen = splitValue * sizeof(float);
    perTokenScaleParams.blockCount = 1;
    perTokenScaleParams.srcStride = 0;
    perTokenScaleParams.dstStride = 0;
    LocalTensor<float> perTokenScaleLocal =
        perTokenScaleInQueue.AllocTensor<float>();
    DataCopyPad(perTokenScaleLocal,
                perTokenScaleGM[mmOutOffset + singleMId * baseM],
                perTokenScaleParams, padParams);
    perTokenScaleInQueue.EnQue(perTokenScaleLocal);
    perTokenScaleInUb = perTokenScaleInQueue.DeQue<float>();
    const uint32_t broadCastDst[2] = {splitValue, this->baseN};
    const uint32_t broadCastSrc[2] = {splitValue, 1};
    BroadCast<float, 2, 1>(pertokenBrcbLocal, perTokenScaleInUb, broadCastDst,
                           broadCastSrc, sharedTmpLocal);
}

__aicore__ inline void KernelMatmulInt128Mlt512::CopycGm(uint32_t mmOutOffset, uint32_t offsetN,
                                 uint32_t splitValue, uint32_t singleMId) {
    LocalTensor mmOutLocal = vecInQueue.AllocTensor<int32_t>();
    DataCopyParams mmOutParams;
    mmOutParams.blockCount = splitValue;
    mmOutParams.blockLen = (this->baseN * sizeof(int32_t)) / 32;
    mmOutParams.srcStride =
        (this->N * sizeof(int32_t) - this->baseN * sizeof(int32_t)) / 32;
    mmOutParams.dstStride = 0;
    uint32_t cGmOffset =
        (mmOutOffset + singleMId * baseM) * N + offsetN * this->baseN;
    AscendC::DataCopy(mmOutLocal, cGM[cGmOffset], mmOutParams);
    vecInQueue.EnQue(mmOutLocal);
}

__aicore__ inline void KernelMatmulInt128Mlt512::CopyScale(uint32_t groupIdx, uint32_t offsetN) {
    LocalTensor<bfloat16_t> scaleLocal = scaleInQueue.AllocTensor<bfloat16_t>();
    uint32_t scaleGmOffset = groupIdx * N + offsetN * this->baseN;
    AscendC::DataCopy(scaleLocal, scaleGM[scaleGmOffset], scaleSize);
    scaleInQueue.EnQue(scaleLocal);
}

// ==================== Implementation of matrix multiplication calculation related functions ====================

__aicore__ inline void KernelMatmulInt128Mlt512::MMCompute(uint32_t groupIdx, uint32_t splitValue,
                                   int32_t *tokensIdx, uint32_t oneTimesData,
                                   uint32_t preOffset) {
    for (size_t singleMId = 0; singleMId < this->roundM; singleMId++) {
      CopyA(baseM, preOffset, tokensIdx, singleMId);
      for (size_t singleNId = 0; singleNId < this->roundN; singleNId++) {
        CopyIn(singleMId, 0, singleNId, groupIdx, splitValue, tokensIdx, oneTimesData, oneTimesData);
        uint32_t tail = this->roundK % oneTimesData;
        uint32_t tilingNums = this->roundK / oneTimesData;
        AscendC::LocalTensor<int32_t> c1Local = outQueueCO1.AllocTensor<int32_t>();
        for (size_t tilingNumId = 0; tilingNumId < tilingNums; tilingNumId++) {
          uint32_t offsetK = tilingNumId * oneTimesData;
          bLocal = inQueueB1.DeQue<int8_t>();
          if (tilingNumId < tilingNums - 1)
            CopyIn(singleMId, tilingNumId + 1, singleNId, groupIdx, splitValue, tokensIdx, oneTimesData, oneTimesData);

          for (size_t singleKId = 0; singleKId < oneTimesData; singleKId++) {
            SplitA(singleMId, singleKId + offsetK, singleNId);
            SplitB(singleMId, singleKId, singleNId);
            Compute(singleMId, singleKId + offsetK, singleNId, c1Local);
          }
          inQueueB1.FreeTensor(bLocal);
        }
        if (tail != 0) {
          CopyIn(singleMId, tilingNums, singleNId, groupIdx, splitValue, tokensIdx, oneTimesData, tail);
          bLocal = inQueueB1.DeQue<int8_t>();
          for (size_t singleKId = 0; singleKId < tail; singleKId++) {
            //  CopyIn(singleMId, singleKId,
            //  singleNId,groupIdx,splitValue,tokensIdx);
            SplitA(singleMId, singleKId + tilingNums * oneTimesData, singleNId);
            SplitB(singleMId, singleKId, singleNId);
            Compute(singleMId, singleKId + tilingNums * oneTimesData, singleNId, c1Local);
          }
          inQueueB1.FreeTensor(bLocal);
        }

        outQueueCO1.EnQue<int32_t>(c1Local);
        CopyOut(singleMId, singleNId, preOffset, oneTimesData);
      }
      inQueueA1.FreeTensor(aLocal);
    }
}

__aicore__ inline void KernelMatmulInt128Mlt512::CopyA(uint32_t splitValue, uint32_t preOffset,
                               int32_t *tokensIdx, uint32_t singleMId) {
    if (this->tailM != 0 && singleMId == this->roundM - 1) {
      splitValue = this->tailM;
    }
    AscendC::LocalTensor<int8_t> a1Local = inQueueA1.AllocTensor<int8_t>();
    AscendC::Nd2NzParams dataCopyA1Params;
    dataCopyA1Params.ndNum = 1;
    dataCopyA1Params.nValue = splitValue;
    dataCopyA1Params.dValue = K;
    dataCopyA1Params.srcNdMatrixStride = 0;
    dataCopyA1Params.srcDValue = K;
    dataCopyA1Params.dstNzC0Stride = baseM;
    dataCopyA1Params.dstNzNStride = 1;
    dataCopyA1Params.dstNzMatrixStride = 1;
    AscendC::DataCopy(a1Local, aGM[(preOffset + singleMId * baseM) * K], dataCopyA1Params);
    inQueueA1.EnQue(a1Local);
    aLocal = inQueueA1.DeQue<int8_t>();
}

__aicore__ inline void KernelMatmulInt128Mlt512::CopyIn(uint32_t singleMId, uint32_t tilingNumId,
                                uint32_t singleNId, uint32_t groupIdx,
                                uint32_t splitValue, int32_t *tokensIdx,
                                uint32_t oneTimesData, uint32_t numbers) {
    AscendC::LocalTensor<int8_t> b1Local = inQueueB1.AllocTensor<int8_t>();
    for(int i=0;i<numbers;i++){
      AscendC::DataCopy(b1Local[i*baseK*baseN], bGM[groupIdx * K * N + singleNId * this->roundK * baseK * baseN + tilingNumId * oneTimesData * baseK * 32 + i*baseK*32], {8,128,static_cast<uint16_t>(K-baseK),0});
    }
    inQueueB1.EnQue(b1Local);
}

__aicore__ inline void KernelMatmulInt128Mlt512::CopyOut(uint32_t singleMId, uint32_t singleNId,
                                 uint32_t preOffset, uint32_t oneTimesData) {
    AscendC::LocalTensor<int32_t> c1Local = outQueueCO1.DeQue<int32_t>();
    AscendC::FixpipeParamsV220 fixpipeParams; // For 910B, this item should be
                                              // replaced with FixpipeParamsV220
    if (this->roundN - 1 == singleNId && tailN != 0) {
      fixpipeParams.nSize = tailN;
    } else {
      fixpipeParams.nSize = baseN;
    }
    if (this->roundM - 1 == singleMId && tailM != 0) {
      fixpipeParams.mSize = tailM;
    } else {
      fixpipeParams.mSize = baseM;
    }

    fixpipeParams.srcStride = baseM;
    fixpipeParams.dstStride = N;

    fixpipeParams.ndNum = 1;
    fixpipeParams.srcNdStride = 0;
    fixpipeParams.dstNdStride = 0;
    auto gmIndex = preOffset * N + singleMId * N * baseM + singleNId * baseN;
    AscendC::Fixpipe(cGM[gmIndex], c1Local, fixpipeParams);

    outQueueCO1.FreeTensor(c1Local);
}


__aicore__ inline void KernelMatmulInt128Mlt512::SplitA(uint32_t singleMId, uint32_t singleKId,
                                uint32_t singleNId) {

    AscendC::LocalTensor<int8_t> a1Local = aLocal[singleKId * baseM * baseK];
    a1Local.SetSize(baseK * baseM);
    AscendC::LocalTensor<int8_t> a2Local = inQueueA2.AllocTensor<int8_t>();
    uint32_t srcOffset = 0;
    uint32_t dstOffset = 0;

    // transform Nz to zZ
    for (uint32_t i = 0; i < mblocks; ++i) {
      AscendC::LoadData2dParams loadL0AParams;
      loadL0AParams.repeatTimes = kblocks;
      loadL0AParams.srcStride = mblocks;
      loadL0AParams.ifTranspose = false;

      AscendC::LoadData(a2Local[dstOffset], a1Local[srcOffset], loadL0AParams);

      srcOffset += 16 * 32;
      dstOffset += baseK * 16;
    }
    inQueueA2.EnQue<int8_t>(a2Local);
}

__aicore__ inline void KernelMatmulInt128Mlt512::SplitB(uint32_t singleMId, uint32_t singleKId,
                                uint32_t singleNId) {
    AscendC::LocalTensor<int8_t> b1Local = bLocal[baseK * baseN * singleKId];
    b1Local.SetSize(baseK * baseN);
    AscendC::LocalTensor<int8_t> b2Local = inQueueB2.AllocTensor<int8_t>();
    uint32_t srcOffset = 0;
    uint32_t dstOffset = 0;
    for (uint32_t i = 0; i < kblocks; ++i) {
      AscendC::LoadData2dTransposeParams loadL0BParams;
      loadL0BParams.repeatTimes = baseN / 32;
      loadL0BParams.srcStride = kblocks;
      loadL0BParams.dstGap = 1;
      loadL0BParams.dstFracGap = 0;

      AscendC::LoadDataWithTranspose(b2Local[dstOffset], b1Local[srcOffset],
                                     loadL0BParams);

      srcOffset += 32 * 32;
      dstOffset += baseN * 32;
    }
    
    inQueueB2.EnQue<int8_t>(b2Local);
}

__aicore__ inline void KernelMatmulInt128Mlt512::Compute(uint32_t singleMId, uint32_t singleKId,
                                 uint32_t singleNId,
                                 AscendC::LocalTensor<int32_t> c1Local) {
    AscendC::LocalTensor<int8_t> a2Local = inQueueA2.DeQue<int8_t>();
    AscendC::LocalTensor<int8_t> b2Local = inQueueB2.DeQue<int8_t>();

    AscendC::MmadParams mmadParams;

    mmadParams.m = baseM;
    mmadParams.k = baseK;
    mmadParams.n = baseN;
    // true: The initial value of the C matrix is ​​0, false: The initial
    // value of the C matrix is ​​configured through the cmatrixSource
    // parameter.
    mmadParams.cmatrixSource = false; // True takes bias from C2, false takes bias from C01
    if (singleKId == 0) {
      mmadParams.cmatrixInitVal = true;
      AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
    } else {
      // If it is not the first block, start accumulating in L0C
      mmadParams.cmatrixInitVal = false;
      AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
    }
    inQueueA2.FreeTensor(a2Local);
    inQueueB2.FreeTensor(b2Local);
}

// ==================== Tool function implementation ====================

__aicore__ inline uint32_t KernelMatmulInt128Mlt512::AlignUp(uint32_t a, uint32_t base) {
    return (a + base - 1) / base * base;
}
template <typename T>
__aicore__ inline __gm__ T* KernelMatmulInt128Mlt512::GetTensorAddr(GM_ADDR tensorPtr) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr));
}
