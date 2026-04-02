/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMBINE_SCALE_HPP
 #define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMBINE_SCALE_HPP
 
 #include "catlass/catlass.hpp"
 #include "catlass/arch/resource.hpp"
 #include "catlass/epilogue/dispatch_policy.hpp"
 #include "catlass/epilogue/tile/tile_copy.hpp"
 #include "catlass/gemm_coord.hpp"
 #include "catlass/matrix_coord.hpp"
 
 namespace Catlass::Epilogue::Block {
 
 template <class OutputType_, class InputType_>
 class BlockEpilogue<EpilogueAtlasA2CombineScale, OutputType_, InputType_> {
 public:
     // Type aliases
     using DispatchPolicy = EpilogueAtlasA2CombineScale;
     using ArchTag = typename DispatchPolicy::ArchTag; 
 
     using ElementOutput = typename OutputType_::Element;
     using ElementInput = typename InputType_::Element;
 
     using LayoutOutput = typename OutputType_::Layout; 
     using LayoutInput = typename InputType_::Layout;
 
        // 常量定义
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t MAX_UB_ELEM_NUM = 32 * 1024 / sizeof(float); // output使用32;
    static constexpr uint32_t SUM_MAX_UB_PINGPONG_OFFSET = 2 * 1024;
    static constexpr uint32_t ATTENTION_TEMP_PINGPONG_OFFSET = 32 * 1024;
    static constexpr uint32_t ATTENTION_TEMP_OFFSET = 2 * ATTENTION_TEMP_PINGPONG_OFFSET; // 64k
    static constexpr uint32_t ATTEN_OUT_OFFSET = 128 * 1024;
    static constexpr uint32_t SOFTMAX_BROAD_SIZE = 8;
     CATLASS_DEVICE
     BlockEpilogue(Arch::Resource<ArchTag> &resource)
     {
         // Allocate UB space
        /*
            shared_gl: 4kb  [2kb ping, 2kb pong]
            unshared_gl: 4kb 
            shared_gm: 4kb
            unshared_gm: 4kb
            shared_out: 64kb [32kb ping, 32kb pong]
            unshared_out: 64kb
            middle Tenosr : 192 - 16 - 128 = 48kb
        */
        sharedOutUbTensor = resource.ubBuf.template GetBufferByByte<float>(0); // 0 - 64k
        unsharedOutUbTensor = resource.ubBuf.template GetBufferByByte<float>(ATTENTION_TEMP_OFFSET); // 64k - 128k
        sharedGlUbTensor = resource.ubBuf.template GetBufferByByte<float>(ATTEN_OUT_OFFSET); // 128k - 132k
        unsharedGlUbTensor = resource.ubBuf.template GetBufferByByte<float>(ATTEN_OUT_OFFSET + 2 * SUM_MAX_UB_PINGPONG_OFFSET); // 132k - 136k
        sharedGmUbTensor = resource.ubBuf.template GetBufferByByte<float>(ATTEN_OUT_OFFSET + 4 * SUM_MAX_UB_PINGPONG_OFFSET); // 136k - 140k
        unsharedGmUbTensor = resource.ubBuf.template GetBufferByByte<float>(ATTEN_OUT_OFFSET + 6 * SUM_MAX_UB_PINGPONG_OFFSET); // 140k - 144k
        realGmUbTensor = resource.ubBuf.template GetBufferByByte<float>(ATTEN_OUT_OFFSET + 8 * SUM_MAX_UB_PINGPONG_OFFSET); // 144k - 148k
        realGlUbTensor = resource.ubBuf.template GetBufferByByte<float>(ATTEN_OUT_OFFSET + 10 * SUM_MAX_UB_PINGPONG_OFFSET); // 148k - 152k
        outUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(ATTEN_OUT_OFFSET + 12 * SUM_MAX_UB_PINGPONG_OFFSET); // 152k - 184k
    }
 
     CATLASS_DEVICE
     ~BlockEpilogue() {}
 
     CATLASS_DEVICE
     void SetMask(int32_t len)
     {
         uint64_t mask = 0;
         uint64_t one = 1;
         uint64_t temp = len % FLOAT_VECTOR_SIZE;
         for (int64_t i = 0; i < temp; i++) {
             mask |= one << i;
         }
 
         if (len == VECTOR_SIZE) {
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
         } else if (len >= FLOAT_VECTOR_SIZE) {
             AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
         } else {
             AscendC::SetVectorMask<int8_t>(0x0, mask);
         }
     }
 
     CATLASS_DEVICE
     void operator()(
         AscendC::GlobalTensor<float> gSharedGm,
         AscendC::GlobalTensor<float> gUnsharedGm,
         AscendC::GlobalTensor<float> gSharedGl,
         AscendC::GlobalTensor<float> gUnsharedGl,
         AscendC::GlobalTensor<float> gSharedOut,
         AscendC::GlobalTensor<float> gUnsharedOut,
         AscendC::GlobalTensor<ElementOutput> gFinalOutput,
         const MatrixCoord &actualBlockShape)
     {
         // 伪代码实现：
         // 1. 计算分块参数
         uint32_t rowNum = actualBlockShape.row();
         uint32_t columnNum = actualBlockShape.column();
         uint32_t columnNumRound = RoundUp(columnNum, FLOAT_BLOCK_SIZE);
         uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
         uint32_t subBlockNum = AscendC::GetSubBlockNum();
         uint32_t rowSplitSubBlock = rowNum / subBlockNum;
         uint32_t rowActualThisSubBlock = (subBlockIdx == 0) ? rowSplitSubBlock : (rowNum - rowSplitSubBlock);
         uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;
         uint32_t rowNumTile = RoundDown(MAX_UB_ELEM_NUM / columnNum, FLOAT_BLOCK_SIZE);
         uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
         uint32_t preLoad = 1;

         if (rowActualThisSubBlock == 0) {
            return;
         }

         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        //  if (subBlockIdx == 1) 
        //     return;
         // 4. 主循环 - preload 和 pingpong 优化
         for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
             if (rowLoopIdx < rowLoopNum) {
                 // 数据加载阶段
                 uint32_t pingpongFlag = rowLoopIdx % 2;
                 uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                 uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                 uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ? 
                     (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                 uint32_t rowNumCurLoopRound = RoundUp(rowNumCurLoop, FLOAT_BLOCK_SIZE);
                 uint32_t pingPongOffset = pingpongFlag * SUM_MAX_UB_PINGPONG_OFFSET / sizeof(float);
                 uint32_t attnPingPongOffset = pingpongFlag * ATTENTION_TEMP_PINGPONG_OFFSET / sizeof(float);
                 uint32_t sumMaxOffsetIoGm = rowOffsetIoGm;
                 uint32_t sumMaxOffsetIoShared = rowOffsetIoGm * SOFTMAX_BROAD_SIZE;
                 uint32_t attnOffsetIoGm = rowOffsetIoGm * columnNum;

                 
                 AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(pingpongFlag);
                 AscendC::DataCopy(sharedGmUbTensor[pingPongOffset], 
                    gSharedGm[sumMaxOffsetIoShared], rowNumCurLoop * SOFTMAX_BROAD_SIZE);
                    AscendC::DataCopy(unsharedGmUbTensor[pingPongOffset], 
                    gUnsharedGm[sumMaxOffsetIoGm], rowNumCurLoopRound); 
                 // Copy GL
                AscendC::DataCopy(sharedGlUbTensor[pingPongOffset], 
                    gSharedGl[sumMaxOffsetIoShared], rowNumCurLoop * SOFTMAX_BROAD_SIZE);
                AscendC::DataCopy(unsharedGlUbTensor[pingPongOffset], 
                    gUnsharedGl[sumMaxOffsetIoGm], rowNumCurLoopRound);

                 // Copy Shared and Unshared
                 // [rowNum, column] 
                 AscendC::DataCopy(sharedOutUbTensor[attnPingPongOffset], 
                    gSharedOut[attnOffsetIoGm], rowNumCurLoop * columnNum);  // 这里一定32B对齐
                 AscendC::DataCopy(unsharedOutUbTensor[attnPingPongOffset], 
                    gUnsharedOut[attnOffsetIoGm], rowNumCurLoop * columnNum);
             }
             
             if (rowLoopIdx >= preLoad) {
                 // 计算阶段
                 uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                 uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                 uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                 uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                 uint32_t rowNumCurLoop = (delayedRowLoopIdx == rowLoopNum - 1) ? 
                     (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                 uint32_t rowNumCurLoopRound = RoundUp(rowNumCurLoop, FLOAT_BLOCK_SIZE);
                 uint32_t pingPongOffset = pingpongFlag * SUM_MAX_UB_PINGPONG_OFFSET / sizeof(float);
                 uint32_t attnPingPongOffset = pingpongFlag * ATTENTION_TEMP_PINGPONG_OFFSET / sizeof(float);
                 // 调用 SubCoreCompute 进行核心计算
                 uint32_t attnOffsetIoGm = rowOffsetIoGm * columnNum;
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
                SubCoreCompute(
                    sharedGmUbTensor[pingPongOffset],
                    unsharedGmUbTensor[pingPongOffset],
                    realGmUbTensor[pingPongOffset],
                    sharedGlUbTensor[pingPongOffset],
                    unsharedGlUbTensor[pingPongOffset],
                    realGlUbTensor[pingPongOffset],
                    sharedOutUbTensor[attnPingPongOffset],
                    unsharedOutUbTensor[attnPingPongOffset],
                    pingpongFlag, rowNumCurLoop, 
                    rowNumCurLoopRound, columnNum, columnNumRound);
                 // 6. Cast 到目标精度 (FP16/BF16)
                 if constexpr (std::is_same_v<ElementOutput, bfloat16_t>) {
                     AscendC::Cast<ElementOutput, float, false>(
                         outUbTensor.template ReinterpretCast<bfloat16_t>()[attnPingPongOffset],
                         sharedOutUbTensor[attnPingPongOffset],
                         AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                         (rowNumCurLoop * columnNumRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                         AscendC::UnaryRepeatParams(1, 1, 4, 8));
                 } else {
                     AscendC::Cast<ElementOutput, float, false>(
                         outUbTensor.template ReinterpretCast<half>()[attnPingPongOffset],
                         sharedOutUbTensor[attnPingPongOffset],
                         AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                         (rowNumCurLoop * columnNumRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                         AscendC::UnaryRepeatParams(1, 1, 4, 8));
                 }
                 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                 AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                 if constexpr (std::is_same_v<ElementOutput, bfloat16_t>) {
                    AscendC::DataCopy(gFinalOutput[attnOffsetIoGm],
                    outUbTensor.template ReinterpretCast<bfloat16_t>()[attnPingPongOffset], 
                    rowNumCurLoop * columnNum);
                 } else {
                    AscendC::GlobalTensor<ElementOutput> gFinal = gFinalOutput[attnOffsetIoGm];
                    AscendC::DataCopy(gFinal,
                    outUbTensor.template ReinterpretCast<half>()[attnPingPongOffset], rowNumCurLoop * columnNum);
                 }
                 AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(pingpongFlag);
             }
         }

         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
     }
 
private:
    // SubCoreCompute 函数：执行核心计算逻辑
    CATLASS_DEVICE
    void SubCoreCompute(
        AscendC::LocalTensor<float> sharedGmUbLoopTensor,
        AscendC::LocalTensor<float> unsharedGmUbLoopTensor,
        AscendC::LocalTensor<float> realGmUbLoopTensor,
        AscendC::LocalTensor<float> sharedGlUbLoopTensor,
        AscendC::LocalTensor<float> unsharedGlUbLoopTensor,
        AscendC::LocalTensor<float> realGlUbLoopTensor,
        AscendC::LocalTensor<float> sharedOutUbLoopTensor,
        AscendC::LocalTensor<float> unsharedOutUbLoopTensor,
        uint32_t pingpongFlag, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, 
                       uint32_t columnNum, uint32_t columnNumRound) {
        // 1. gm = max(shared_gm, unshared_gm)
        // sharedGmUbLoopTensor [row, 8]
        // unsharedGmUbLoopTensor [row, 1]
        AscendC::BlockReduceMax<float, false>(
            sharedGmUbLoopTensor,
            sharedGmUbLoopTensor,
            rowNumCurLoop,
            uint64_t(0),
            1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::BlockReduceMax<float, false>(
            sharedGlUbLoopTensor,
            sharedGlUbLoopTensor,
            rowNumCurLoop,
            uint64_t(0),
            1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Max<float>(
            realGmUbLoopTensor,
            sharedGmUbLoopTensor,
            unsharedGmUbLoopTensor,
            rowNumCurLoop);
        AscendC::PipeBarrier<PIPE_V>();
        
        // 2. update_shared_expgm = exp(shared_gm - gm)
        //    update_unshared_expgm = exp(unshared_gm - gm)
        AscendC::Sub<float>(
            sharedGmUbLoopTensor, sharedGmUbLoopTensor,
            realGmUbLoopTensor, rowNumCurLoop);
        AscendC::Sub<float>(
               unsharedGmUbLoopTensor,
               unsharedGmUbLoopTensor,
               realGmUbLoopTensor,
               rowNumCurLoop);
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::Exp<float>(
            sharedGmUbLoopTensor,
            sharedGmUbLoopTensor,
            rowNumCurLoop);
       AscendC::Exp<float>(
               unsharedGmUbLoopTensor,
               unsharedGmUbLoopTensor,
               rowNumCurLoop);
        AscendC::PipeBarrier<PIPE_V>();
        
        // 3. update_unshared_expgm = exp(unshared_gm - gm)
        
        // 4. gl = shared_gl * update_shared_expgm + unshared_gl * update_unshared_expgm
        // 先计算 shared_gl * update_shared_expgm
        AscendC::Mul<float>(
           sharedGlUbLoopTensor,
            sharedGlUbLoopTensor,
            sharedGmUbLoopTensor,
            rowNumCurLoop);

        // 计算 unshared_gl * update_unshared_expgm 并累加到 gl
        AscendC::Mul<float>(
            unsharedGlUbLoopTensor,
            unsharedGlUbLoopTensor,
            unsharedGmUbLoopTensor,
            rowNumCurLoop);
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::Add<float>(
            realGlUbLoopTensor,
            sharedGlUbLoopTensor,
            unsharedGlUbLoopTensor,
            rowNumCurLoop);
        AscendC::PipeBarrier<PIPE_V>();
        

        // BroadCast update_shared_expgm and update_unshared_expgm
        AscendC::Brcb(sharedGmUbLoopTensor,
           sharedGmUbLoopTensor,
           rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
           AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(unsharedGmUbLoopTensor,
           unsharedGmUbLoopTensor,
           rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
           AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        
        // 5. final_true_out = (shared_true_out * update_shared_expgm + unshared_true_out * update_unshared_expgm) / gl
        // 计算 shared_true_out * update_shared_expgm
        for (uint32_t mulIdx = 0; mulIdx < columnNum / FLOAT_VECTOR_SIZE; ++mulIdx) {
            AscendC::Mul<float, false>(
                sharedOutUbLoopTensor[mulIdx * FLOAT_VECTOR_SIZE],
                sharedOutUbLoopTensor[mulIdx * FLOAT_VECTOR_SIZE],
                sharedGmUbLoopTensor,
                AscendC::MASK_PLACEHOLDER, rowNumCurLoop,
                AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SetMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Mul<float, false>(
                sharedOutUbLoopTensor[columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                sharedOutUbLoopTensor[columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                sharedGmUbLoopTensor,
                AscendC::MASK_PLACEHOLDER, rowNumCurLoop,
                AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t mulIdx = 0; mulIdx < columnNum / FLOAT_VECTOR_SIZE; ++mulIdx) {
            AscendC::Mul<float, false>(
                unsharedOutUbLoopTensor[mulIdx * FLOAT_VECTOR_SIZE],
                unsharedOutUbLoopTensor[mulIdx * FLOAT_VECTOR_SIZE],
                unsharedGmUbLoopTensor,
                (uint64_t)0, rowNumCurLoop,
                AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SetMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Mul<float, false>(
                unsharedOutUbLoopTensor[columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                unsharedOutUbLoopTensor[columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                unsharedGmUbLoopTensor,
                (uint64_t)0, rowNumCurLoop,
                AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        
        // 累加到 sharedOutUbTensor
        for (uint32_t addIdx = 0; addIdx < columnNum / FLOAT_VECTOR_SIZE; ++addIdx) {
            AscendC::Add<float, false>(
                sharedOutUbLoopTensor[addIdx * FLOAT_VECTOR_SIZE],
                sharedOutUbLoopTensor[addIdx * FLOAT_VECTOR_SIZE],
                unsharedOutUbLoopTensor[addIdx * FLOAT_VECTOR_SIZE],
                (uint64_t)0, rowNumCurLoop,
                AscendC::BinaryRepeatParams(1, 1, 1, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE));
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SetMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Add<float, false>(
                sharedOutUbLoopTensor[columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                sharedOutUbLoopTensor[columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                unsharedOutUbLoopTensor[columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                (uint64_t)0, rowNumCurLoop,
                AscendC::BinaryRepeatParams(1, 1, 1, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        
        // 除以 gl
        AscendC::Brcb(realGlUbLoopTensor,
           realGlUbLoopTensor,
           rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
           AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t divIdx = 0; divIdx < columnNum / FLOAT_VECTOR_SIZE; ++divIdx) {
            AscendC::Div<float, false>(
                sharedOutUbLoopTensor[divIdx * FLOAT_VECTOR_SIZE],
                sharedOutUbLoopTensor[divIdx * FLOAT_VECTOR_SIZE],
                realGlUbLoopTensor,
                (uint64_t)0, rowNumCurLoop,
                AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SetMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Div<float, false>(
                sharedOutUbLoopTensor[columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                sharedOutUbLoopTensor[columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                realGlUbLoopTensor,
                (uint64_t)0, rowNumCurLoop,
                AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // UB 张量定义
    AscendC::LocalTensor<float> sharedGmUbTensor;
    AscendC::LocalTensor<float> unsharedGmUbTensor;
    AscendC::LocalTensor<float> realGmUbTensor;
    AscendC::LocalTensor<float> realGlUbTensor;
    AscendC::LocalTensor<float> sharedGlUbTensor;
    AscendC::LocalTensor<float> unsharedGlUbTensor;
    AscendC::LocalTensor<float> sharedOutUbTensor;
    AscendC::LocalTensor<float> unsharedOutUbTensor;
    AscendC::LocalTensor<ElementOutput> outUbTensor;

 };
 
 } // namespace Catlass::Epilogue::Block
 
 #endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMBINE_SCALE_HPP