#ifndef INCLUDE_UTILS_H
#define INCLUDE_UTILS_H

template <typename IN_DTYPE>
__aicore__ inline void CreateCaMatrix(const AscendC::LocalTensor<IN_DTYPE> &dst,
                                      const uint16_t repeats,
                                      const uint16_t blockNum,
                                      const uint16_t dstGap,
                                      const IN_DTYPE initValue)
{
    AscendC::InitConstValue<IN_DTYPE>(dst,
                                      AscendC::InitConstValueParams<IN_DTYPE>(repeats, blockNum, dstGap, initValue));
}

__aicore__ inline void SetFftsBaseAddr(uint64_t config)
{
    AscendC::SetSyncBaseAddr(config);
}

template <typename IN_DTYPE>
__aicore__ inline void SetPadding(IN_DTYPE padValue)
{
    AscendC::SetLoadDataPaddingValue<IN_DTYPE>(padValue);
}

__aicore__ inline void SetAtomicnone()
{
    AscendC::SetAtomicNone();
}

__aicore__ inline void SetMasknorm()
{
#if __CCE_AICORE__ == 100
    return;
#endif
    AscendC::SetMaskNorm();
}

__aicore__ inline void SetNdpara(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
{
    AscendC::SetFixpipeNz2ndFlag(ndNum, srcNdStride, dstNdStride);
}

template <typename IN_DTYPE>
__aicore__ inline void SetVectorMask(const uint64_t maskHigh, const uint64_t maskLow)
{
    AscendC::SetVectorMask<IN_DTYPE>(maskHigh, maskLow);
}

__aicore__ inline int64_t GetSubBlockidx()
{
    return AscendC::GetSubBlockIdx();
}

__aicore__ inline void WaitFlagDev(uint16_t flagId)
{
    AscendC::WaitEvent(flagId);
}

template <pipe_t pipe, uint8_t mode>
__aicore__ inline void FftsCrossCoreSync(uint16_t flagId)
{
    AscendC::CrossCoreSetFlag<mode, pipe>(flagId);
}

template <typename IN_DTYPE, bool setRelu = false>
__aicore__ inline void SetFpc(const AscendC::LocalTensor<IN_DTYPE> &preTensor, bool isUnitFlag = false)
{
    AscendC::SetFixPipeConfig<IN_DTYPE, setRelu>(preTensor, isUnitFlag);
}

template <typename IN_DTYPE>
__aicore__ inline void CopyCbufToFbuf(AscendC::LocalTensor<IN_DTYPE> &dst,
                                      AscendC::LocalTensor<IN_DTYPE> &src,
                                      uint16_t burstNum,
                                      uint16_t burstLen,
                                      uint16_t srcGapSize,
                                      uint16_t dstGapSize)
{
    dst.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2PIPE2GM);
    AscendC::DataCopy(dst,
                      src,
                      AscendC::DataCopyParams(burstNum,     // nBurst
                                              burstLen,     // lenBurst
                                              srcGapSize,   // srcGap
                                              dstGapSize)); // dstGap);
}

template <typename IN_DTYPE>
__aicore__ inline void CopyCbufToBt(uint64_t dst,
                                    const AscendC::LocalTensor<IN_DTYPE> &src,
                                    uint16_t convControl,
                                    uint16_t nBurst,
                                    uint16_t lenBurst,
                                    uint16_t sourceGap,
                                    uint16_t dstGap)
{
    AscendC::LocalTensor<IN_DTYPE> dstTensor;
    dstTensor.InitBuffer(dst, nBurst * lenBurst);
    dstTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2);
    AscendC::DataCopy(dstTensor,
                      src,
                      AscendC::DataCopyParams(nBurst,    // nBurst
                                              lenBurst,  // lenBurst
                                              sourceGap, // srcGap
                                              dstGap));  // dstGap);
}
#endif