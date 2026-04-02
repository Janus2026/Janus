/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_MMAD_PV_SPLIT_ROW_HPP
#define CATLASS_GEMM_BLOCK_MMAD_PV_SPLIT_ROW_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

////////////////////////////////////////////////////////////////////

namespace Catlass::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <
    bool PAGED_CACHE_FLAG_,
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmad<
    MmadAtlasA2FAIPVSplitRow<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2FAIPVSplitRow<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1A_SIZE = 32768 * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    static_assert(L1TileShape::M * L1TileShape::K <= 32768, "L1TileShape::M * L1TileShape::K must be less than 32768");

    /// Construct
    CATLASS_DEVICE
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        // Allocate L1 memory space
        l1BTensor = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1A_SIZE * 2);
        for (uint32_t i = 0; i < STAGES; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + L1A_SIZE * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmad()
    {}

    CATLASS_DEVICE
    void getKVOffset(AscendC::GlobalTensor<int32_t> &gBlockTable, uint32_t &kOffset, uint32_t &nowNIdx,
        uint32_t &strideKV, uint32_t &blockSize)
    {
        if constexpr (PAGED_CACHE_FLAG_) {
            uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);
            kOffset = blockTableId * blockSize * strideKV;
        } else {
            kOffset = nowNIdx * blockSize * strideKV;
        }
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementB> gB,
        AscendC::GlobalTensor<ElementC> gC, AscendC::GlobalTensor<int32_t> gBlockTable, LayoutA layoutA,
        LayoutB layoutB, LayoutC layoutC, GemmCoord actualOriShape, uint32_t &nIdx, uint32_t &blockSize,
        uint32_t strideKV, Arch::CrossCoreFlag softmaxFlag, uint32_t &l1KPPingPongFlag, uint32_t &l0ABPingPongFlag,
        uint32_t &l0CPingPongFlag)
    {
        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        uint32_t stackSeqTile = actualOriShape[2];
        uint32_t gBOffset = 0;

        // load V
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(stackSeqTile, embed);
        uint32_t kLoop = CeilDiv(stackSeqTile, blockSize);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        for (uint32_t blockStackIdx = 0; blockStackIdx < kLoop; blockStackIdx++) {
            uint32_t nowNIdx = nIdx + blockStackIdx;
            uint32_t kActual = AscendC::Std::min(stackSeqTile - blockStackIdx * blockSize, blockSize);
            getKVOffset(gBlockTable, gBOffset, nowNIdx, strideKV, blockSize);

            auto layoutBTile = layoutB.GetTileLayout(MakeCoord(kActual, embed));
            MatrixCoord l1BTileCoord{blockStackIdx * blockSize, 0};
            auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];
            copyGmToL1B(l1BTile, gB[gBOffset], layoutBInL1, layoutBTile);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);

        Arch::CrossCoreWaitFlag(softmaxFlag);

        uint32_t mL1Loop = CeilDiv<L1TileShape::M>(rowNum);
        uint32_t kL1Loop = CeilDiv<L1TileShape::K>(stackSeqTile);
        for (uint32_t mL1Idx = 0; mL1Idx < mL1Loop; mL1Idx++) {
            uint32_t mL1Actual = (mL1Idx < mL1Loop - 1) ? L1TileShape::M : (rowNum - mL1Idx * L1TileShape::M);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
            for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; kL1Idx++) {
                uint32_t kL1Actual = (kL1Idx < kL1Loop - 1) ? L1TileShape::K : (stackSeqTile - kL1Idx * L1TileShape::K);

                // load P
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KPPingPongFlag);
                MatrixCoord gmATileCoord{mL1Idx * L1TileShape::M, kL1Idx * L1TileShape::K};
                auto gmTileA = gA[layoutA.GetOffset(gmATileCoord)];
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(mL1Actual, kL1Actual));
                LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(mL1Actual, kL1Actual);
                copyGmToL1A(l1ATensor[l1KPPingPongFlag], gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KPPingPongFlag);

                uint32_t kL0Loop = CeilDiv<L0TileShape::K>(kL1Actual);
                for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    uint32_t kL0Actual =
                        (kL0Idx < kL0Loop - 1) ? L0TileShape::K : (kL1Actual - kL0Idx * L0TileShape::K);

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, embed);
                    MatrixCoord l1BTileCoord{kL1Idx * L1TileShape::K + kL0Idx * L0TileShape::K, 0};
                    auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];

                    copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTile, layoutBInL0, layoutBInL1);

                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL1Actual, kL0Actual);
                    MatrixCoord l1ATileCoord{0, kL0Idx * L0TileShape::K};
                    auto l1ATile = l1ATensor[l1KPPingPongFlag][layoutAInL1.GetOffset(l1ATileCoord)];

                    if (kL0Idx == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KPPingPongFlag);
                    }
                    copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATile, layoutAInL0, layoutAInL1);
                    if (kL0Idx == kL0Loop - 1) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KPPingPongFlag);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag);
                    bool initMmad = kL1Idx == 0 && kL0Idx == 0;
                    tileMmad(l0CTensor[l0CPingPongFlag],
                        l0ATensor[l0ABPingPongFlag],
                        l0BTensor[l0ABPingPongFlag],
                        mL1Actual,
                        embed,
                        kL0Actual,
                        initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    l0ABPingPongFlag = 1 - l0ABPingPongFlag;
                }
                l1KPPingPongFlag = 1 - l1KPPingPongFlag;
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CPingPongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CPingPongFlag);
            MatrixCoord gmCTileCoord{mL1Idx * L0TileShape::M, 0};
            LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL1Actual, embed));
            auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL1Actual, embed));
            copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPongFlag], layoutCTile, layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
            l0CPingPongFlag = 1 - l0CPingPongFlag;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor;
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

////////////////////////////////////////////////////////////////////

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_BLOCK_MMAD_PV_SPLIT_ROW_HPP