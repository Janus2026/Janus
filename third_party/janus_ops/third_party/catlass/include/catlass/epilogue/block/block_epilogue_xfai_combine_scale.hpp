
#ifndef CATLASS_EPILOGUE_BLOCK_XFAI_COMBINE_SCALE_HPP
#define CATLASS_EPILOGUE_BLOCK_XFAI_COMBINE_SCALE_HPP

#include <limits>
#include "catlass/arch/resource.hpp"

namespace Catlass::Epilogue::Block {

template <
    class OutputType_,
    class LseType_>
class CombineScale {
public:
    using ElementOutput = typename OutputType_::Element;
    using ElementLse = typename LseType_::Element;
    using ArchTag = Arch::AtlasA2;

    // Placeholder constants matching original logic
    static constexpr uint32_t STAGE2_UB_UINT8_BLOCK_SIZE = 6144; // 24 * 64 * 4
    static constexpr uint32_t UB_UINT8_LINE_SIZE = 32768; // 1 * 64 * 128 * 4
    static constexpr uint32_t BASIC_BLOCK = 64 * 128;

    __aicore__ inline CombineScale(Arch::Resource<ArchTag> &resource) {
        // UB Memory Allocation
        constexpr uint32_t LL_UB_OFFSET = 0; // splitnum_align * (q * h)_algin
        constexpr uint32_t LM_UB_OFFSET = 1 * STAGE2_UB_UINT8_BLOCK_SIZE;  // 1 * (q * h)_algin
        constexpr uint32_t BROADCAST_OFFSET = 2 * STAGE2_UB_UINT8_BLOCK_SIZE; // splitnum_align * (q * h)_algin
        constexpr uint32_t TL_UB_OFFSET = 3 * STAGE2_UB_UINT8_BLOCK_SIZE; // splitnum_align * (q * h)_algin
        constexpr uint32_t RS_UB_OFFSET = 4 * STAGE2_UB_UINT8_BLOCK_SIZE; // 1 * (q * h)_algin
        constexpr uint32_t TS_UB_OFFSET = 5 * STAGE2_UB_UINT8_BLOCK_SIZE; // 1 * (q * h)_algin
        constexpr uint32_t BROADCASTSCALE_OFFSET = 6 * STAGE2_UB_UINT8_BLOCK_SIZE; // splitnum_align * (q * h)_algin
        constexpr uint32_t GL_UB_OFFSET = 7 * STAGE2_UB_UINT8_BLOCK_SIZE; // splitnum_align * (q * h)_algin
        constexpr uint32_t BROADCASTO_OFFSET = 8 * STAGE2_UB_UINT8_BLOCK_SIZE; //splitnum_align * (q * h)_algin * v
        constexpr uint32_t GO16_UB_OFFSET = 8 * STAGE2_UB_UINT8_BLOCK_SIZE + 2 * UB_UINT8_LINE_SIZE; // (q * h)_algin * v
        constexpr uint32_t tempReduceMax_OFFSET = 9 * STAGE2_UB_UINT8_BLOCK_SIZE; //splitnum_align * (q * h)_algin * v
        constexpr uint32_t tempReduceSum_OFFSET = 10 * STAGE2_UB_UINT8_BLOCK_SIZE; //splitnum_align * (q * h)_algin * v
        constexpr uint32_t GO_UB_OFFSET = 0; // (q * h)_algin * v
        
        // Buffer Init
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_OFFSET);
        broadCastTensor = resource.ubBuf.template GetBufferByByte<float>(BROADCAST_OFFSET);
        tlUbTensor = resource.ubBuf.template GetBufferByByte<float>(TL_UB_OFFSET);
        rsUbTensor = resource.ubBuf.template GetBufferByByte<float>(RS_UB_OFFSET);
        tsUbTensor = resource.ubBuf.template GetBufferByByte<float>(TS_UB_OFFSET);
        broadCastScaleTensor = resource.ubBuf.template GetBufferByByte<float>(BROADCASTSCALE_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_OFFSET);
        broadCastOTensor = resource.ubBuf.template GetBufferByByte<float>(BROADCASTO_OFFSET);
        toUbTensor = resource.ubBuf.template GetBufferByByte<float>(BROADCASTO_OFFSET);
        goUbTensor = resource.ubBuf.template GetBufferByByte<float>(GO_UB_OFFSET);
        loFloatUbTensor = resource.ubBuf.template GetBufferByByte<float>(GO16_UB_OFFSET);
        go16UbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_UB_OFFSET);

        // Temp buffers - allocating from remaining space or overlapping where safe
        // Adjust offsets as needed for temp buffers
        //这两块空间看情况
        tempReduceMax = resource.ubBuf.template GetBufferByByte<uint8_t>(tempReduceMax_OFFSET);
        tempReduceSum = resource.ubBuf.template GetBufferByByte<uint8_t>(tempReduceSum_OFFSET); // Offset example

    }


    __aicore__ inline void operator()(
        uint32_t qHeads,
        uint32_t kvSplitCoreNum,
        uint32_t headSizeV,
        __gm__ SplitKvExtraInfo *extraInfo,
        AscendC::GlobalTensor<ElementLse> lGmTensor,
        AscendC::GlobalTensor<ElementLse> oCoreTmpGmTensor,
        AscendC::GlobalTensor<ElementOutput> oGmTensor,
        AscendC::GlobalTensor<int32_t> gActualQseqlen,
        bool inputLayoutTND = true
    ) {

        int64_t vectorsubBlockID = AscendC::GetSubBlockIdx();
        int64_t subBlockNum = AscendC::GetBlockNum() * 2;
        int64_t subBlockID = AscendC::GetBlockIdx();

        for (uint32_t process = subBlockID; process < kvSplitCoreNum * 2; process += subBlockNum) {
            uint32_t batchIdx = extraInfo->splitInfo[process/2].batchIdx;
            uint32_t headStartIndx = extraInfo->splitInfo[process/2].headStartIdx;
            uint32_t headEndIndx = extraInfo->splitInfo[process/2].headEndIdx;
            uint32_t qStartIndx = extraInfo->splitInfo[process/2].qStartIdx;
            uint32_t qEndIndx = extraInfo->splitInfo[process/2].qEndIdx;
            uint32_t pingpongFlag = 0;

            uint32_t q_len = (qEndIndx - qStartIndx);
            uint32_t n_len = (headEndIndx - headStartIndx);

            uint32_t sum = q_len * n_len;
            uint32_t sum_former = q_len == 1 ? sum / 2 : (q_len / 2) * n_len;

            uint32_t addrLOffset = vectorsubBlockID == 0 ? extraInfo->splitInfo[process/2].lseTaskOffset : extraInfo->splitInfo[process/2].lseTaskOffset+ sum_former;
            uint32_t addrOOffset = vectorsubBlockID == 0 ? extraInfo->splitInfo[process/2].oTaskOffset : extraInfo->splitInfo[process/2].oTaskOffset + sum_former * headSizeV;

            uint32_t prevQSeqlenSum = 0;
            if (inputLayoutTND) {
                prevQSeqlenSum = (batchIdx == 0) ?
                    0 : static_cast<uint32_t>(gActualQseqlen.GetValue(batchIdx - 1));
            }
            uint32_t baseGmOffset = prevQSeqlenSum * qHeads * headSizeV + qStartIndx * qHeads * headSizeV + headStartIndx * headSizeV;
            uint32_t gmOScalar = 0;
            if (q_len == 1) {
                gmOScalar = vectorsubBlockID == 0 ? baseGmOffset 
                                                : baseGmOffset + sum_former * headSizeV;
            } else {
                uint32_t q_half = q_len / 2;
                gmOScalar = vectorsubBlockID == 0 ? baseGmOffset 
                                                : baseGmOffset + q_half * qHeads * headSizeV;
            }

            uint32_t splitNum = extraInfo->splitInfo[process/2].splitNum;

            uint32_t splitNumAlign = (splitNum + 7) / 8 * 8;  // 32b align
            uint32_t lseBlock = vectorsubBlockID == 0 ? sum_former : sum - sum_former;
            uint32_t lseBlockAlign = (lseBlock + 7) / 8 * 8;  // 32b align
            int32_t count = splitNum * lseBlockAlign;
            int32_t lnCount = 1 * lseBlockAlign;
            // Initialize LSE UB space
            int32_t calcLen = splitNumAlign * lseBlockAlign;
            int32_t oCount = lseBlock * headSizeV;
            int32_t lseCount = lseBlockAlign * headSizeV;
            int32_t oCount_vector = sum * headSizeV;
            
            AscendC::Duplicate(llUbTensor, std::numeric_limits<float>::lowest(), calcLen);
            AscendC::Duplicate(tlUbTensor, 0.0f, calcLen);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

            // Copy LSE from GM to UB
            uint32_t srcStride = vectorsubBlockID == 0 ? sum - sum_former : sum_former;
            AscendC::DataCopyPad(llUbTensor, lGmTensor[addrLOffset], 
                                AscendC::DataCopyExtParams(splitNum, lseBlock * sizeof(float), srcStride * sizeof(float), 0, 0),
                                AscendC::DataCopyPadExtParams<float>(false, 0, lseBlockAlign - lseBlock, 0));

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            // // Preload O
            AscendC::DataCopyPad(loFloatUbTensor[pingpongFlag * BASIC_BLOCK], oCoreTmpGmTensor[addrOOffset], 
                AscendC::DataCopyExtParams(1, oCount * sizeof(float), 0, 0, 0),
                AscendC::DataCopyPadExtParams<float>(true, 0, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);

            // ReduceMax
            uint32_t reduceMaxShape[] = { splitNumAlign, lseBlockAlign };
            AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA, false>(lmUbTensor, llUbTensor, tempReduceMax, reduceMaxShape, true);
            AscendC::PipeBarrier<PIPE_V>();

            // Broadcast Max
            uint32_t dstShapeBroadcast[] = { splitNum, lseBlockAlign };
            uint32_t srcShapeBroadcast[] = { 1, lseBlockAlign };
            AscendC::BroadCast<float, 2, 0>(broadCastTensor, lmUbTensor, dstShapeBroadcast, srcShapeBroadcast, tempReduceSum);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Sub(tlUbTensor, llUbTensor, broadCastTensor, count);
            AscendC::PipeBarrier<PIPE_V>();

            // expf
            AscendC::Exp(tlUbTensor, tlUbTensor, count);
            AscendC::PipeBarrier<PIPE_V>();

            // ReduceSum
            uint32_t reduceSumShape[] = { splitNumAlign, lseBlockAlign };
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(rsUbTensor, tlUbTensor, tempReduceSum, reduceSumShape, true);
            AscendC::PipeBarrier<PIPE_V>();

            // Ln
            AscendC::Ln(rsUbTensor, rsUbTensor, lnCount);
            AscendC::PipeBarrier<PIPE_V>();

            // logf(lse_sum) + lse_max
            AscendC::Add(tsUbTensor, rsUbTensor, lmUbTensor, lnCount);
            AscendC::PipeBarrier<PIPE_V>();

            // Broadcast scale
            AscendC::BroadCast<float, 2, 0>(broadCastScaleTensor, tsUbTensor, dstShapeBroadcast, srcShapeBroadcast, tempReduceSum);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Sub(glUbTensor, llUbTensor, broadCastScaleTensor, count);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Exp(glUbTensor, glUbTensor, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            for (uint32_t nIdx = 0; nIdx < splitNum; nIdx++) {
                if (nIdx != 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
                    AscendC::DataCopyPad(loFloatUbTensor[pingpongFlag * BASIC_BLOCK], oCoreTmpGmTensor[addrOOffset + nIdx * oCount_vector], 
                        AscendC::DataCopyExtParams(1, oCount * sizeof(float), 0, 0, 0),
                        AscendC::DataCopyPadExtParams<float>(true, 0, 0, 0));
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);

                uint32_t dstShapeO[2] = { lseBlockAlign, headSizeV };
                uint32_t srcShapeO[2] = { lseBlockAlign, 1 };
                AscendC::BroadCast<float, 2, 1>(broadCastOTensor[pingpongFlag * BASIC_BLOCK], glUbTensor[nIdx * lseBlockAlign], dstShapeO, srcShapeO, tempReduceSum);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Mul(toUbTensor[pingpongFlag * BASIC_BLOCK], loFloatUbTensor[pingpongFlag * BASIC_BLOCK], broadCastOTensor[pingpongFlag * BASIC_BLOCK], oCount); // toUbTensor和broadCastOTensor共用一块空间
                AscendC::PipeBarrier<PIPE_V>();

                if (nIdx == 0) {
                    AscendC::Adds(goUbTensor, toUbTensor[pingpongFlag * BASIC_BLOCK], 0.0f, oCount); // TODO goUbTensor和loFloatUbTensor一块空间
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    AscendC::Add(goUbTensor, toUbTensor[pingpongFlag * BASIC_BLOCK], goUbTensor, oCount);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
                pingpongFlag = 1 - pingpongFlag;
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::PipeBarrier<PIPE_V>();
            // Cast and move out
            if (std::is_same<ElementOutput, bfloat16_t>::value) {
                AscendC::Cast(go16UbTensor, goUbTensor, AscendC::RoundMode::CAST_RINT, oCount);
            } else {
                AscendC::Cast(go16UbTensor, goUbTensor, AscendC::RoundMode::CAST_NONE, oCount);
            }
            
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            
            if (q_len == 1) {
                AscendC::DataCopyPad(oGmTensor[gmOScalar], go16UbTensor, AscendC::DataCopyExtParams(1, oCount * sizeof(ElementOutput) , 0, 0, 0));
            } else {
                uint32_t q_half = q_len / 2;
                if (vectorsubBlockID == 0) {
                    AscendC::DataCopyPad(oGmTensor[gmOScalar], go16UbTensor,
                            AscendC::DataCopyExtParams(q_half, (headEndIndx - headStartIndx) * headSizeV * sizeof(ElementOutput) , 0, (qHeads - (headEndIndx - headStartIndx)) * headSizeV * sizeof(ElementOutput), 0));
                } else {
                    AscendC::DataCopyPad(oGmTensor[gmOScalar], go16UbTensor,
                            AscendC::DataCopyExtParams(q_len - q_half, (headEndIndx - headStartIndx) * headSizeV * sizeof(ElementOutput) , 0, (qHeads - (headEndIndx - headStartIndx)) * headSizeV * sizeof(ElementOutput), 0));
                }
            }
        }
    }

private:
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<float> lmUbTensor;
    AscendC::LocalTensor<float> tlUbTensor;
    AscendC::LocalTensor<float> rsUbTensor;
    AscendC::LocalTensor<float> tsUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> toUbTensor;
    AscendC::LocalTensor<float> goUbTensor;
    AscendC::LocalTensor<ElementOutput> go16UbTensor;
    AscendC::LocalTensor<float> loFloatUbTensor;

    AscendC::LocalTensor<uint8_t> tempReduceMax;
    AscendC::LocalTensor<uint8_t> tempReduceSum;
    AscendC::LocalTensor<float> broadCastTensor;
    AscendC::LocalTensor<float> broadCastScaleTensor;
    AscendC::LocalTensor<float> broadCastOTensor;
};
}

#endif // CATLASS_EPILOGUE_BLOCK_XFAI_COMBINE_SCALE_HPP

